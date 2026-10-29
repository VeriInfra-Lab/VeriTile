import bench.repairs.tritonbench_g.triton_matmul.Proofs.Wrapper
import bench.tritonbench_g.triton_matmul.Proofs.LogicalViewDefect
import bench.tritonbench_g.matmul_triton_autotune.Proofs.Real
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
open VeriTile.Triton FP.Structural FPValue
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 4000000

def idealNumerics : Numerics ℝ where
  zero := 0
  loadZero := 0
  dotAcc := fun k a b acc => acc+∑ j : Fin k, a j*b j
  castOutput := id
abbrev ideal := LogicalViewDefect.ideal
abbrev idealDot := LogicalViewDefect.idealDot

theorem ideal_primitives (st : Storage) : PrimitiveLaws ideal idealDot idealNumerics st := by
  cases st <;> constructor
  all_goals intros
  all_goals simp [ideal,LogicalViewDefect.ideal,idealNumerics,
    idealDot,LogicalViewDefect.idealDot,LogicalViewDefect.idealDotValue,dtype]
  all_goals norm_num

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
  rw [MatmulTritonAutotune.Whole.foldRange_sum _ 0 _ 0 (Nat.zero_le _)]
  simp only [Finset.range_zero,Finset.sum_empty,zero_add,sub_zero]
  rw [MatmulTritonAutotune.Whole.sum_blocks]
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

theorem resolved_mathematical (r : Request) (before prepared : State ℝ) (tensor : TensorAPI r prepared)
    (A : Fin r.rows → Fin r.inner → ℝ) (B : Fin r.inner → Fin r.columns → ℝ)
    (resolve : ResolveNegContract r before prepared tensor A B) :
    ∃ final, host ideal idealDot r prepared=.ok (r.C,final) ∧
      (∀ i : Fin r.rows, ∀ j : Fin r.columns,
        final.mem r.C (outputOffset r i.val j.val)=stored r.storage (SourceScope.mathematical r.rows r.columns r.inner A B i j)) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=prepared.mem region offset) ∧
      (∀ region offset, ¬Footprint r region offset → ¬CopyFootprint tensor resolve.copyA resolve.copyB region offset →
        final.mem region offset=before.mem region offset) ∧
      (∀ offset, final.mem resolve.originalA offset=before.mem resolve.originalA offset) ∧
      (∀ offset, final.mem resolve.originalB offset=before.mem resolve.originalB offset) ∧
      AddressSafety r prepared tensor := by
  obtain ⟨final,he,⟨hv,hf⟩,hc,hA,hB,hs⟩ := resolved_host ideal idealDot idealNumerics r
    (ideal_primitives r.storage) before prepared tensor A B resolve
  refine ⟨final,he,?_,hf,hc,hA,hB,hs⟩
  intro i j
  simpa only [ideal_reference r prepared A B resolve.logicalA resolve.logicalB i j]
    using hv i.val i.isLt j.val j.isLt

/-- Complete source-ordered wrapper and independent mathematical GEMM. -/
theorem wrapper_mathematical (api : WrapperAPI ℝ) (r : Request) (before : State ℝ)
    (A : Fin r.rows → Fin r.inner → ℝ) (B : Fin r.inner → Fin r.columns → ℝ)
    (apiLaw : WrapperAPIContract api r before A B) :
    match wrapper ideal idealDot api r before with
    | .error e => WrapperFailure api r before e
    | .ok (c,final) => WrapperSuccess idealNumerics api r before A B c final ∧
        (∀ i : Fin r.rows, ∀ j : Fin r.columns,
          final.mem r.C (outputOffset r i.val j.val)=stored r.storage (SourceScope.mathematical r.rows r.columns r.inner A B i j)) := by
  have outcome := wrapper_correct ideal idealDot idealNumerics api r (ideal_primitives r.storage) before A B apiLaw
  cases hw : wrapper ideal idealDot api r before with
  | error e => simpa only [WrapperOutcome,hw] using outcome
  | ok output =>
    rcases output with ⟨c,final⟩
    have success : WrapperSuccess idealNumerics api r before A B c final := by
      simpa only [WrapperOutcome,hw] using outcome
    refine ⟨success,?_⟩
    obtain ⟨_,prepared,tensor,resolve,_,result,_⟩ := success
    intro i j
    have value := result.1 i.val i.isLt j.val j.isLt
    rw [ideal_reference r prepared A B resolve.logicalA resolve.logicalB i j] at value
    exact value

#axiomsClean wrapper_mathematical

#axiomsClean ideal_primitives
#axiomsClean resolved_mathematical
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
