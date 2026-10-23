import bench.tritonbench_g.matmul_triton_autotune.Proofs.WholeTask
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel

/-! Exact mathematical projection of the complete original host/grid.
This is the ideal Real model. Native FP32 dot rounding is covered separately
by FPEquiv's exact primitive interface; no Real=FP32 assumption is made here.
-/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Whole
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 4000000

def idealNumerics (roundHalf : ℝ → ℝ) : Numerics ℝ where
  zero := 0
  dotAcc := fun width a b acc => acc + ∑ k : Fin width, a k*b k
  nonnegative := fun x => decide (0≤x)
  slopeMul := fun x => (1/100 : ℝ)*x
  castHalf := roundHalf

def idealAlgebra (roundHalf : ℝ → ℝ) : Algebra ℝ :=
  { ContextAttnFwd.FPModel.model with
    cast := fun _ _ target x => if target=.fp16 then roundHalf x else x }

def idealDotValue {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → ℝ) (b : TileIndex (batch++[k,n]) → ℝ) :
    TileIndex (batch++[m,n]) → ℝ :=
  match batch with
  | [] => fun i => ∑ j : Fin k, a (i.1,j,PUnit.unit)*b (j,i.2.1,PUnit.unit)
  | _::tail => fun i => idealDotValue (batch:=tail) (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2
termination_by batch.length

def idealDot : FPValue.DotAcc ℝ where
  apply := fun _ _ _ _ _ {batch} {m} {k} {n} a b acc i => acc i + idealDotValue (batch:=batch) a b i
  le := fun _ _ x y => decide (x≤y)

theorem ideal_primitives (roundHalf : ℝ → ℝ) :
    PrimitiveLaws (idealAlgebra roundHalf) idealDot (idealNumerics roundHalf) := by
  constructor
  all_goals intros
  all_goals simp [idealAlgebra,ContextAttnFwd.FPModel.model,idealNumerics,idealDot,idealDotValue]
  all_goals norm_num

theorem foldRange_sum (f : Nat → ℝ) (start stop : Nat) (a : ℝ) (h : start≤stop) :
    foldRange (fun k x => x+f k) start stop 1 a=
      a+(∑ k∈Finset.range stop,f k)-(∑ k∈Finset.range start,f k) := by
  by_cases hs : start<stop
  · rw [foldRange]
    simp only [Nat.one_ne_zero,hs,↓reduceDIte]
    rw [foldRange_sum f (start+1) stop (a+f start) (by omega),Finset.sum_range_succ]
    ring
  · have he : start=stop := by omega
    subst stop
    rw [foldRange]
    simp
termination_by stop-start
decreasing_by omega

theorem sum_blocks (f : Nat → ℝ) (blocks width : Nat) :
    (∑ b∈Finset.range blocks, ∑ k : Fin width,f (b*width+k.val))=
      ∑ k∈Finset.range (blocks*width),f k := by
  induction blocks with
  | zero => simp
  | succ b ih =>
    rw [Finset.sum_range_succ,ih,Nat.succ_mul,Finset.sum_range_add]
    rw [Fin.sum_univ_eq_sum_range (fun k => f (b*width+k)) width]

theorem ideal_accumulator (roundHalf : ℝ → ℝ) (r : Request) (initial : State ℝ) (row col : Nat) :
    accumulator (idealNumerics roundHalf) r initial row col=
      ∑ k : Fin r.inner,inputA r initial row k.val*inputB r initial k.val col := by
  let f := fun k => if k<r.inner then inputA r initial row k*inputB r initial k col else 0
  have hb (block : Nat) (acc : ℝ) : blockStep (idealNumerics roundHalf) r initial row col block acc=
      acc + ∑ k : Fin r.config.bk,f (block*r.config.bk+k.val) := by
    simp only [blockStep,idealNumerics]
    congr 1
    apply Finset.sum_congr rfl
    intro j _
    dsimp only [f]
    split_ifs <;> simp
  unfold accumulator
  change foldRange (fun block x => blockStep (idealNumerics roundHalf) r initial row col block x)
    0 ((r.inner+r.config.bk-1)/r.config.bk) 1 0=_
  simp_rw [hb]
  rw [foldRange_sum _ 0 _ 0 (Nat.zero_le _)]
  simp only [Finset.range_zero,Finset.sum_empty,zero_add,sub_zero]
  rw [sum_blocks]
  have hbound : r.inner≤((r.inner+r.config.bk-1)/r.config.bk)*r.config.bk := by cases hh : r.config <;> simp [hh,Config.bk] <;> omega
  have htruncate : (∑k∈Finset.range (((r.inner+r.config.bk-1)/r.config.bk)*r.config.bk),f k)=∑k∈Finset.range r.inner,f k := by
    symm
    apply Finset.sum_subset (Finset.range_mono hbound)
    intro k _ hk
    simp only [Finset.mem_range] at hk
    simp [f,hk]
  rw [htruncate,Finset.sum_range]
  apply Finset.sum_congr rfl
  intro k _
  simp [f,k.isLt]

theorem ideal_reference (roundHalf : ℝ → ℝ) (r : Request) (initial : State ℝ)
    (A : Fin r.rows → Fin r.inner → ℝ) (B : Fin r.inner → Fin r.columns → ℝ)
    (hA : ∀ i k,inputA r initial i.val k.val=A i k)
    (hB : ∀ k j,inputB r initial k.val j.val=B k j)
    (i : Fin r.rows) (j : Fin r.columns) :
    reference (idealNumerics roundHalf) r initial i.val j.val=roundHalf (mathematical r A B i j) := by
  unfold reference
  rw [ideal_accumulator]
  have hs : (∑ k : Fin r.inner,inputA r initial i.val k.val*inputB r initial k.val j.val)=
      ∑ k : Fin r.inner,A i k*B k j := by
    apply Finset.sum_congr rfl
    intro k _
    rw [hA,hB]
  rw [hs]
  simp [Numerics.activate,idealNumerics,mathematical]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Whole
