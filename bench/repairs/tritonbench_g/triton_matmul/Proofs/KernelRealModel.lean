import bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelExecution
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelRealModel
open VeriTile.Triton FP.Structural FPValue
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 4000000

/-- Task-local ideal Real interpretation, constructed against the current
Algebra structure. It does not import the stale historical FPModel backend.
Only literal, cast and the explicit dotAcc backend are reached by this kernel. -/
def ideal : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a+b | .sub => a-b | .mul => a*b | .div => a/b
    | .max => max a b | .pow => 0
  fma := fun _ => some (fun a b c => a*b+c)
  atomicAdd := fun _ => some (· + ·)
  unary := fun _ _ _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0
  scan := fun _ {_} _ _ _ _ => none
  dot := fun _ {_} {_} {_} {_} _ _ => none

def idealDotValue {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → ℝ) (b : TileIndex (batch++[k,n]) → ℝ) :
    TileIndex (batch++[m,n]) → ℝ :=
  match batch with
  | [] => fun i => ∑ j : Fin k,a (i.1,j,PUnit.unit)*b (j,i.2.1,PUnit.unit)
  | _::tail => fun i => idealDotValue (batch:=tail) (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2
termination_by batch.length

def idealDot : DotAcc ℝ where
  apply := fun _ _ _ _ _ {batch} {m} {k} {n} a b acc i => acc i+idealDotValue (batch:=batch) a b i

def idealNumerics : Numerics ℝ where
  zero := 0
  loadZero := 0
  dotAcc := fun k a b acc => acc+∑ j : Fin k,a j*b j
  castOutput := id

theorem cast_identity (x : ℝ) : ideal.cast none .real .fp16 x=x := rfl

theorem ideal_primitives (st : Storage) : PrimitiveLaws ideal idealDot idealNumerics st := by
  cases st <;> constructor
  all_goals intros
  all_goals simp [ideal,idealNumerics,idealDot,idealDotValue,dtype]
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

theorem ideal_accumulator (r : Request) (initial : State ℝ) (row col : Nat) :
    accumulator idealNumerics r initial row col=
      ∑ k : Fin r.inner,inputA r initial row k.val*inputB r initial k.val col := by
  let f := fun k => if k<r.inner then inputA r initial row k*inputB r initial k col else 0
  have hb (block : Nat) (acc : ℝ) : blockStep idealNumerics r initial row col block acc=
      acc + ∑ k : Fin (SourceScope.blockK r.storage),f (block*(SourceScope.blockK r.storage)+k.val) := by
    simp only [blockStep,idealNumerics]
    congr 1
    apply Finset.sum_congr rfl
    intro j _
    dsimp only [f]
    split_ifs <;> simp
  unfold accumulator
  change foldRange (fun block x => blockStep idealNumerics r initial row col block x)
    0 ((r.inner+(SourceScope.blockK r.storage)-1)/(SourceScope.blockK r.storage)) 1 0=_
  simp_rw [hb]
  rw [foldRange_sum _ 0 _ 0 (Nat.zero_le _)]
  simp only [Finset.range_zero,Finset.sum_empty,zero_add,sub_zero]
  rw [sum_blocks]
  have hbound : r.inner≤((r.inner+(SourceScope.blockK r.storage)-1)/(SourceScope.blockK r.storage))*(SourceScope.blockK r.storage) := by cases hh : r.storage <;> simp [hh,SourceScope.blockK] <;> omega
  have htruncate : (∑k∈Finset.range (((r.inner+(SourceScope.blockK r.storage)-1)/(SourceScope.blockK r.storage))*(SourceScope.blockK r.storage)),f k)=∑k∈Finset.range r.inner,f k := by
    symm
    apply Finset.sum_subset (Finset.range_mono hbound)
    intro k _ hk
    simp only [Finset.mem_range] at hk
    simp [f,hk]
  rw [htruncate,Finset.sum_range]
  apply Finset.sum_congr rfl
  intro k _
  simp [f,k.isLt]

theorem ideal_reference (r : Request) (initial : State ℝ)
    (A : Fin r.rows → Fin r.inner → ℝ) (B : Fin r.inner → Fin r.columns → ℝ)
    (hA : ∀ i k,inputA r initial i.val k.val=A i k)
    (hB : ∀ k j,inputB r initial k.val j.val=B k j)
    (i : Fin r.rows) (j : Fin r.columns) :
    reference idealNumerics r initial i.val j.val=SourceScope.mathematical r.rows r.columns r.inner A B i j := by
  unfold reference
  rw [ideal_accumulator]
  have hs : (∑ k : Fin r.inner,inputA r initial i.val k.val*inputB r initial k.val j.val)=
      ∑ k : Fin r.inner,A i k*B k j := by
    apply Finset.sum_congr rfl
    intro k _
    rw [hA,hB]
  rw [hs]
  rfl


#axiomsClean cast_identity
#axiomsClean ideal_primitives
#axiomsClean foldRange_sum
#axiomsClean sum_blocks
#axiomsClean ideal_accumulator
#axiomsClean ideal_reference
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelRealModel
