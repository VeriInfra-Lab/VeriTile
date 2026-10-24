import bench.repairs.tritonbench_g.seeded_dropout.Contract
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SeededDropout
open VeriTile.Torch
noncomputable section

/-- Specialize the shared Tensor contract to the same input in both slots;
this drops reflexive shape/device checks without importing host arithmetic. -/
private theorem tensor_contract_iff (x : TensorMetadata) : TensorContract x ↔
    x.layout = .strided ∧ x.dtype = .floating .fp32 ∧ x.device.kind = "cuda" ∧
    x.isContiguous = Bool.true ∧ x.isNeg = Bool.false ∧ x.numel < 2^31 ∧
    (0 < x.numel → x.dataPtr.toNat % 4 = 0) := by
  constructor
  · intro h
    exact ⟨h.layout.1,h.dtype.1,h.cuda.1,h.contiguous.1,h.noNeg.1,h.extent,
      fun hn => (h.alignment hn).1⟩
  · rintro ⟨hl,hd,hc,ht,hn,he,ha⟩
    exact ⟨⟨hl,hl⟩,⟨hd,hd⟩,⟨hc,hc⟩,rfl,rfl,⟨ht,ht⟩,⟨hn,hn⟩,he,
      fun hp => ⟨ha hp,ha hp⟩⟩

private theorem tensor_check_iff (x : TensorMetadata) :
    AddExample.Contiguous.checkTensorInputs x x = .ok () ↔ TensorContract x := by
  rw [tensor_contract_iff]
  simp [AddExample.Contiguous.checkTensorInputs,TensorMetadata.isCuda,ite_eq_iff]

theorem input_contract_iff (x : TensorMetadata) (p : ℝ) (s : Int) :
    InputContract x p s ↔ TensorContract x ∧ (0 ≤ p ∧ p < 1) ∧ -(2^63 : Int) ≤ s ∧ s < 2^64 := by
  constructor
  · intro h; exact ⟨h.tensor,h.probability,h.seedRange⟩
  · rintro ⟨ht,hp,hs⟩; exact ⟨ht,hp,hs⟩

theorem checkInputs_iff (x : ArgumentMetadata) (p : ProbabilityArgument) (s : SeedArgument) :
    checkInputs x p s = .ok () ↔ AdmittedArguments x p s := by
  cases x with
  | nonTensor => simp [checkInputs,AddExample.Contiguous.checkInputs,AdmittedArguments]
  | tensor x =>
    cases h : AddExample.Contiguous.checkTensorInputs x x with
    | error e =>
      have bad : ¬ TensorContract x := by
        intro good
        have ok := (tensor_check_iff x).mpr good
        rw [h] at ok
        contradiction
      cases p <;> cases s <;>
        simp [checkInputs,AddExample.Contiguous.checkInputs,h,AdmittedArguments,input_contract_iff,bad]
    | ok u =>
      have good : TensorContract x := (tensor_check_iff x).mp h
      cases p <;> cases s <;>
        simp [checkInputs,AddExample.Contiguous.checkInputs,h,checkScalars,AdmittedArguments,
          input_contract_iff,good,ite_eq_iff]

theorem selected_seed (s : Int) (h : -(2^63 : Int) ≤ s ∧ s < 2^64) :
    SeedABI (selectSeed s).1 (selectSeed s).2 ∧ seedValue (selectSeed s).1 (selectSeed s).2 = s := by
  by_cases hi : -(2^31 : Int) ≤ s ∧ s < 2^31
  · have equal : selectSeed s = ⟨.i32,s⟩ := by unfold selectSeed; rw [if_pos hi]
    rw [equal]
    exact ⟨Or.inl ⟨rfl,hi⟩,rfl⟩
  · by_cases hh : s < 2^63
    · have equal : selectSeed s = ⟨.i64,s⟩ := by unfold selectSeed; rw [if_neg hi,if_pos hh]
      rw [equal]
      exact ⟨Or.inr (Or.inl ⟨rfl,h.1,hh,by change s < -(2^31 : Int) ∨ (2^31 : Int) ≤ s; omega⟩),rfl⟩
    · have hn : 0 ≤ s := by omega
      have equal : selectSeed s = ⟨.u64,s.toNat⟩ := by unfold selectSeed; rw [if_neg hi,if_neg hh]
      rw [equal]
      constructor
      · apply Or.inr; apply Or.inr
        change SeedKind.u64 = SeedKind.u64 ∧ (2^63 : Int) ≤ (s.toNat : Int) ∧ (s.toNat : Int) < 2^64
        rw [Int.toNat_of_nonneg hn]
        exact ⟨rfl,by omega,h.2⟩
      · exact Int.toNat_of_nonneg hn

#axiomsClean checkInputs_iff
#axiomsClean selected_seed
end
end VeriTile.Bench.Repairs.SeededDropout
