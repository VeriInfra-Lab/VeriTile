import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Contract
import bench.repairs.tritonbench_g.streamk_matmul.Proofs.Contract

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Contract
open VeriTile.Torch

theorem contract_iff (a b : TensorMetadata) (c : Config) :
    InputContract a b c ↔ Checked.Contract.InputContract a b c ∧ (Checked.Contract.Empty a b ∨ StaticTiles c) := by
  constructor
  · intro h; exact ⟨h.prior,h.static⟩
  · rintro ⟨h,s⟩; exact ⟨h,s⟩

theorem checkTensorInputs_iff (a b : TensorMetadata) (c : Config) :
    checkTensorInputs a b c = .ok () ↔ InputContract a b c := by
  rw [contract_iff,← Checked.Contract.checkTensorInputs_iff]
  by_cases empty : Checked.Contract.Empty a b <;>
    cases h : Checked.Contract.checkTensorInputs a b c <;> simp [checkTensorInputs,h,empty]

theorem checkInputs_iff (a b : ArgumentMetadata) (c : Config) :
    checkInputs a b c = .ok () ↔ AdmittedArguments a b c := by
  cases a <;> cases b <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]

theorem static_nonempty (a b : TensorMetadata) (c : Config)
    (h : InputContract a b c) (nonempty : ¬ Checked.Contract.Empty a b) : StaticTiles c :=
  h.static.resolve_left nonempty

theorem empty_preserved (a b : TensorMetadata) (c : Config)
    (h : Checked.Contract.InputContract a b c) (empty : Checked.Contract.Empty a b) : InputContract a b c :=
  ⟨h,Or.inl empty⟩

example : ¬ StaticTiles ⟨0,16,16,8,Bool.false,3,4⟩ := by decide
example : StaticTiles ⟨0,1,1,16,Bool.false,3,4⟩ := by decide
example : ¬ StaticTiles ⟨0,2048,1024,16,Bool.false,3,4⟩ := by decide
example : ¬ StaticTiles ⟨0,131072,1,16,Bool.false,3,4⟩ := by decide
example : ¬ StaticTiles ⟨0,1,131072,16,Bool.false,3,4⟩ := by decide

#axiomsClean checkInputs_iff
#axiomsClean static_nonempty
#axiomsClean empty_preserved
end VeriTile.Bench.Repairs.StreamK.TileChecked.Contract
