import bench.tritonbench_g.rotary_transform_ops.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.RotaryTransformOps.KernelIndex
open VeriTile.Triton FP.Structural
noncomputable section

def dataBound (s : Shape) := Host.size s+8*s.heads*s.width+256
def coefficientBound (c : Config) (batch : Nat) :=
  (c.shift batch+c.shape.maximum+8)*c.shape.half+256

/-- The unchanged post-preparation formed-pointer guard, stated at the actual
Tensor data pointers. Empty calls do not inspect unused launch pointers. -/
def pointerCheck (c : Config) (tensors : RegionName→KernelTensor.Tensor) : Bool :=
  Host.empty c.shape || decide (
    (tensors "x").dataPtr.toNat+4*dataBound c.shape≤2^64 ∧
    (tensors (KernelAlgebra.regionName c.shape.output)).dataPtr.toNat+4*dataBound c.shape≤2^64 ∧
    ∀ b : Fin c.shape.batches,∀ sine : Bool,
      (tensors (if sine then "sin" else "cos")).dataPtr.toNat+4*coefficientBound c b.val≤2^64)

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

theorem data_pointer (c : Config) (tensors : RegionName→KernelTensor.Tensor)
    (checked : pointerCheck c tensors=Bool.true) (launch : Host.empty c.shape=Bool.false)
    (offset : Nat) (inside : offset<dataBound c.shape) (output : Bool) :
    let r : RegionName := if output then KernelAlgebra.regionName c.shape.output else "x"
    (BitVec.ofNat 64 (tensors r).dataPtr.toNat+BitVec.ofNat 64 offset*BitVec.ofNat 64 4).toNat=
      (tensors r).dataPtr.toNat+4*offset := by
  simp only [pointerCheck,launch,Bool.false_or,decide_eq_true_eq] at checked
  apply pointer_word
  cases output <;> simp only [Bool.false_eq_true,if_false,if_true] <;> omega

theorem coefficient_pointer (c : Config) (tensors : RegionName→KernelTensor.Tensor)
    (checked : pointerCheck c tensors=Bool.true) (launch : Host.empty c.shape=Bool.false)
    (batch : Fin c.shape.batches) (sine : Bool) (offset : Nat)
    (inside : offset<coefficientBound c batch.val) :
    (BitVec.ofNat 64 (tensors (if sine then "sin" else "cos")).dataPtr.toNat+
      BitVec.ofNat 64 offset*BitVec.ofNat 64 4).toNat=
      (tensors (if sine then "sin" else "cos")).dataPtr.toNat+4*offset := by
  simp only [pointerCheck,launch,Bool.false_or,decide_eq_true_eq] at checked
  have bound := checked.2.2 batch sine
  apply pointer_word
  omega

/-- Every padded coefficient expression used by the quotation has the same
signed32 and signed64 interpretation, including masked fallback lanes. -/
theorem padded_coefficient (c : Config) (abi : IndexABI c) (pid : Program)
    (hp : pid∈programs c) (lane col : Nat) (hl : lane<c.shape.blockM) (hk : col<256) :
    let off := (pid.1*c.shape.blockM+lane+c.shift pid.2.1)*c.shape.half+col
    (BitVec.ofNat 32 off).toInt=(off : Int) ∧ (BitVec.ofNat 64 off).toInt=(off : Int) := by
  have bound := padded_coeff_signed32 c abi pid hp lane col hl hk
  exact ⟨signed32 _ bound,signed64 _ bound⟩

#axiomsClean pointer_word
#axiomsClean data_pointer
#axiomsClean coefficient_pointer
#axiomsClean padded_coefficient
end
end VeriTile.Bench.Repairs.RotaryTransformOps.KernelIndex
