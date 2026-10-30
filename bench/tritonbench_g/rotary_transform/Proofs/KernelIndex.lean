import bench.tritonbench_g.rotary_transform.Proofs.KernelPublic

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.KernelIndex
open VeriTile.Bench.Repairs.RotaryTransform
noncomputable section

theorem pointer_word (base offset : Nat) (bound : base+4*offset<2^64) :
    (BitVec.ofNat 64 base+BitVec.ofNat 64 offset*BitVec.ofNat 64 4).toNat=base+4*offset := by
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add]
  simp only [BitVec.toNat_ofNat,Nat.mul_comm offset 4]
  exact Nat.mod_eq_of_lt bound

theorem signed32 (n : Nat) (bound : n<2^31) : (BitVec.ofNat 32 n).toInt=(n : Int) := by
  have word : (BitVec.ofNat 32 n).toNat=n := by
    simp only [BitVec.toNat_ofNat]
    exact Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt,word]
  rw [word]
  norm_num
  omega

theorem signed64 (n : Nat) (bound : n<2^31) : (BitVec.ofNat 64 n).toInt=(n : Int) := by
  have word : (BitVec.ofNat 64 n).toNat=n := by
    simp only [BitVec.toNat_ofNat]
    exact Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt,word]
  rw [word]
  norm_num
  omega

theorem checked_indices (c : Checked.Call) (bytes : Physical.Bytes) (accepted : check c bytes=Bool.true) :
    IndexABI (Checked.configuration c bytes) := Checked.indices c bytes (admitted c bytes accepted)

theorem data_pointer (c : Checked.Call) (out : Binding.Output c) (bytes : Physical.Bytes)
    (checked : Binding.pointerCheck c out bytes=Bool.true) (launch : Host.empty c.host.shape=Bool.false)
    (offset : Nat) (inside : offset<Binding.dataBound c) (output : Bool) :
    let base := if output then out.tensor.base else c.input.base
    (BitVec.ofNat 64 base+BitVec.ofNat 64 offset*BitVec.ofNat 64 4).toNat=base+4*offset := by
  have bound := Binding.pointer_limits c out bytes checked launch offset inside
  apply pointer_word
  cases output <;> simp only [Bool.false_eq_true,if_false,if_true] <;> omega

theorem coefficient_pointer (c : Checked.Call) (out : Binding.Output c) (bytes : Physical.Bytes)
    (checked : Binding.pointerCheck c out bytes=Bool.true) (launch : Host.empty c.host.shape=Bool.false)
    (batch : Fin c.host.shape.batches) (sine : Bool) (offset : Nat)
    (inside : offset<Binding.coefficientBound c bytes batch.val) :
    (BitVec.ofNat 64 (out.coefficientBase sine)+BitVec.ofNat 64 offset*BitVec.ofNat 64 4).toNat=
      out.coefficientBase sine+4*offset := by
  have bound := Binding.coefficient_pointer_limits c out bytes checked launch batch sine offset inside
  exact pointer_word _ _ (by omega)

theorem padded_coefficient (c : Checked.Call) (bytes : Physical.Bytes) (accepted : check c bytes=Bool.true)
    (pid : Program) (hp : pid∈programs (Checked.configuration c bytes)) (lane col : Nat)
    (hl : lane<c.host.shape.blockM) (hk : col<c.host.shape.blockK) :
    let cfg := Checked.configuration c bytes
    let off := (pid.1*c.host.shape.blockM+lane+cfg.shift pid.2.1)*c.host.shape.half+col
    (BitVec.ofNat 32 off).toInt=(off : Int) ∧ (BitVec.ofNat 64 off).toInt=(off : Int) := by
  have bound := padded_coeff_signed32 (Checked.configuration c bytes) (checked_indices c bytes accepted)
    pid hp lane col hl hk
  exact ⟨signed32 _ bound,signed64 _ bound⟩

theorem checked_tiles (c : Checked.Call) (bytes : Physical.Bytes) (accepted : check c bytes=Bool.true)
    (launch : Host.empty c.host.shape=Bool.false) :
    ∀ shape∈SourceTiles.statementListShapes
      (TypedExecution.kernel c.host.shape c.host.abi.cumulative64 c.host.abi.offsets64).surfaceBody,
      shape.prod≤1048576 :=
  SourceTiles.quotation_shapes_bounded _ _ _ (tiles c bytes accepted launch)

#axiomsClean checked_indices
#axiomsClean data_pointer
#axiomsClean coefficient_pointer
#axiomsClean padded_coefficient
#axiomsClean checked_tiles
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.KernelIndex
