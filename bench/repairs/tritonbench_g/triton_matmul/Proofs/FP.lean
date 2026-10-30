import bench.repairs.tritonbench_g.triton_matmul.Proofs.Wrapper
import bench.tritonbench_g.triton_matmul.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
open VeriTile.Triton FP.Structural FPValue
open ParallelAttention.FPStages (foldRange)
noncomputable section
def logicalStep {α : Type} (Q : Numerics α) (r : Request)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (i : Fin r.rows) (j : Fin r.columns) (block : Nat) (acc : α) : α :=
  Q.dotAcc (SourceScope.blockK r.storage)
    (fun k => if hk : block*(SourceScope.blockK r.storage)+k.val<r.inner then A i ⟨block*(SourceScope.blockK r.storage)+k.val,hk⟩ else Q.loadZero)
    (fun k => if hk : block*(SourceScope.blockK r.storage)+k.val<r.inner then B ⟨block*(SourceScope.blockK r.storage)+k.val,hk⟩ j else Q.loadZero) acc

def logicalReference {α : Type} (Q : Numerics α) (r : Request)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (i : Fin r.rows) (j : Fin r.columns) : α :=
  let value := foldRange (fun block x => logicalStep Q r A B i j block x)
    0 r.parameters.numKBlocks 1 Q.zero
  Q.castOutput value

theorem reference_logical {α : Type} [Inhabited α]
    (Q : Numerics α) (r : Request) (initial : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (hA : ∀ i k,inputA r initial i.val k.val=A i k)
    (hB : ∀ k j,inputB r initial k.val j.val=B k j)
    (i : Fin r.rows) (j : Fin r.columns) :
    reference Q r initial i.val j.val=logicalReference Q r A B i j := by
  have hb (block : Nat) (acc : α) : blockStep Q r initial i.val j.val block acc=logicalStep Q r A B i j block acc := by
    unfold blockStep logicalStep
    apply congrArg₂ (fun a b => Q.dotAcc (SourceScope.blockK r.storage) a b acc)
    · funext k
      by_cases hk : block*(SourceScope.blockK r.storage)+k.val<r.inner
      · simp only [hk,if_true,dif_pos]
        exact hA i ⟨block*(SourceScope.blockK r.storage)+k.val,hk⟩
      · simp [hk]
    · funext k
      by_cases hk : block*(SourceScope.blockK r.storage)+k.val<r.inner
      · simp only [hk,if_true,dif_pos]
        exact hB ⟨block*(SourceScope.blockK r.storage)+k.val,hk⟩ j
      · simp [hk]
  unfold reference accumulator logicalReference
  simp_rw [hb]

theorem resolved_fp {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (Q : Numerics α) (r : Request)
    (law : PrimitiveLaws M D Q r.storage) (before prepared : State α) (tensor : TensorAPI r prepared)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (resolve : ResolveNegContract r before prepared tensor A B) :
    ∃ final, host M D r prepared=.ok (r.C,final) ∧
      (∀ i : Fin r.rows, ∀ j : Fin r.columns,
        final.mem r.C (outputOffset r i.val j.val)=stored r.storage (logicalReference Q r A B i j)) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=prepared.mem region offset) ∧
      (∀ region offset, ¬Footprint r region offset → ¬CopyFootprint tensor resolve.copyA resolve.copyB region offset →
        final.mem region offset=before.mem region offset) ∧
      (∀ offset, final.mem resolve.originalA offset=before.mem resolve.originalA offset) ∧
      (∀ offset, final.mem resolve.originalB offset=before.mem resolve.originalB offset) ∧
      AddressSafety r prepared tensor := by
  obtain ⟨final,he,⟨hv,hf⟩,hc,hA,hB,hs⟩ := resolved_host M D Q r law before prepared tensor A B resolve
  refine ⟨final,he,?_,hf,hc,hA,hB,hs⟩
  intro i j
  simpa only [reference_logical Q r prepared A B resolve.logicalA resolve.logicalB i j]
    using hv i.val i.isLt j.val j.isLt

/-- Concrete existing software profile: encoded FP16/E4M3FN operands,
ascending-lane FP32 FMA with live carry, then the actual storage encoding.
This is not a claim about a GPU tensor-core dotAcc implementation. -/
def softwareNumerics (st : Storage) : Numerics Word where
  zero := FP.Value.zero .fp32
  loadZero := encode st (FP.Value.zero .fp32)
  dotAcc := fun k a b acc => (List.finRange k).foldl
    (fun acc j => FP.fma .ieeeValue (decode st (a j)) (decode st (b j)) acc) acc
  castOutput := encode st

theorem software_primitives (rest : Algebra Word) (st : Storage) :
    PrimitiveLaws (software rest) sequentialDotAcc (softwareNumerics st) st := by
  cases st <;> constructor
  all_goals intros
  all_goals simp [softwareNumerics,software,sequentialDotAcc,bitDotAcc,dtype]
  all_goals norm_num

/-- Complete wrapper result with the independent logical FP target. -/
theorem wrapper_fp {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (Q : Numerics α) (api : WrapperAPI α)
    (r : Request) (law : PrimitiveLaws M D Q r.storage) (before : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (apiLaw : WrapperAPIContract api r before A B) :
    match wrapper M D api r before with
    | .error e => WrapperFailure api r before e
    | .ok (c,final) => WrapperSuccess Q api r before A B c final ∧
        (∀ i : Fin r.rows, ∀ j : Fin r.columns,
          final.mem r.C (outputOffset r i.val j.val)=stored r.storage (logicalReference Q r A B i j)) := by
  have outcome := wrapper_correct M D Q api r law before A B apiLaw
  cases hw : wrapper M D api r before with
  | error e => simpa only [WrapperOutcome,hw] using outcome
  | ok output =>
    rcases output with ⟨c,final⟩
    have success : WrapperSuccess Q api r before A B c final := by
      simpa only [WrapperOutcome,hw] using outcome
    refine ⟨success,?_⟩
    obtain ⟨_,prepared,tensor,resolve,_,result,_⟩ := success
    intro i j
    have value := result.1 i.val i.isLt j.val j.isLt
    rw [reference_logical Q r prepared A B resolve.logicalA resolve.logicalB i j] at value
    exact value

#axiomsClean wrapper_fp

#axiomsClean software_primitives
#axiomsClean resolved_fp
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
