import bench.repairs.tritonbench_g.matmul_dequant_int4.FPEquiv
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Witnesses
open Helpers Physical
set_option maxRecDepth 8192

def tiny : Shape := ⟨3,8,24,8⟩
theorem tiny_admitted : tiny.Valid := by constructor <;> decide +kernel
def singletonStrides : Strides.Packed := ⟨⟨8,1⟩,⟨8,1⟩,⟨1,3⟩⟩
theorem singleton_strides_admitted : singletonStrides.Admitted tiny := by
  unfold Strides.Packed.Admitted Strides.Layout.Admitted singletonStrides tiny
  decide +kernel
theorem singleton_guard : Strides.Guard singletonStrides tiny := by constructor <;> decide +kernel
def fusedShape : Shape := ⟨3,256,128,128⟩
def fusedIndex : IndexDomain fusedShape where
  paddedM := 128
  paddedN := 256
  paddedK := 256
  rows := by decide +kernel
  columns := by decide +kernel
  inner := by decide +kernel
  scalar := by decide +kernel
  a := by decide +kernel
  q := by decide +kernel
  scale := by decide +kernel
  zero := by decide +kernel
  output := by decide +kernel
  dequant := by decide +kernel
def fusedDomain : MachineDomain.Fused fusedShape true where
  source := ⟨by constructor <;> decide +kernel,by decide +kernel,by decide +kernel,
    by decide +kernel,by decide +kernel⟩
  indices := fusedIndex
  mEnvelope := by decide +kernel
  kEnvelope := by decide +kernel
  grid := by decide +kernel

theorem fresh_seed_defect : (1:Word)+packLoop (fun _ => (0:BitVec 4)) 8≠packedSpec (fun _ => (0:BitVec 4)) := by decide +kernel
theorem signed_high_nibble : nibble (0xffffffff:Word) 7=15 := by decide +kernel
theorem half_nan_bytes : decode16 (encode16 (0x7e31:Half))=(0x7e31:Half) := decode16_encode16 _
theorem word_nan_bytes : decode32 (encode32 (0x7fc01234:Word))=(0x7fc01234:Word) := decode32_encode32 _
theorem canonical_singleton_rejection_was_real : ((1,1) : Nat × Nat)≠(256,1) := by decide +kernel
theorem standalone_tail_is_masked : ¬(63<tiny.k) ∧ ¬(63<tiny.n) := by decide +kernel
theorem invalid_group_rejected : ¬(Shape.Valid ⟨1,8,8,0⟩) := by
  intro h;have := h.gpos;change 0<0 at this;omega

def zeroNative : Native.Instructions :=
  ⟨(fun _ _ => 0),(fun _ => 0),(fun _ _ => 0),(fun _ _ => 0),(fun _ _ => 0),(fun _ _ _ => 0),
   (fun _ _ => 0),(fun _ _ => 0),(fun _ => 0),(fun _ => 0),(fun _ => 0),
   (fun _ _ => 0),(fun _ => 0),(fun _ _ => 0),(fun _ _ => 0),(fun _ _ => 0),(fun _ _ _ => 0),
   (fun _ _ => 0),(fun _ _ => 0),(fun _ => 0),(fun _ => 0),(fun _ => 0),
   (fun _ _ _ => 0),(fun _ _ _ => 0),(fun _ => 0),(fun _ => 0),(fun _ _ _ _ => 0)⟩
def badScalar : Native.Instructions := {zeroNative with fma16 := fun _ _ _ => 1}
def badISA : Native.Instructions := {zeroNative with mma16 := fun _ _ _ _ => 1}
theorem scalar_numeric_not_automatic : ¬Native.Correspondence zeroNative badScalar := by
  intro h
  have he := h.fma16 0 0 0
  change (0:Half)=1 at he
  exact (by decide : (0:Half)≠1) he
theorem isa_numeric_not_automatic : ¬Native.Correspondence zeroNative badISA := by
  intro h
  have he := h.mma16 .mma16x8 (fun _ => 0) (fun _ => 0) 0
  change (0:Word)=1 at he
  exact (by decide : (0:Word)≠1) he
theorem native_consistency : WholeNative zeroNative zeroNative := whole_native _

#axiomsClean tiny_admitted
#axiomsClean singleton_strides_admitted
#axiomsClean singleton_guard
#axiomsClean fusedDomain
#axiomsClean fresh_seed_defect
#axiomsClean signed_high_nibble
#axiomsClean half_nan_bytes
#axiomsClean word_nan_bytes
#axiomsClean canonical_singleton_rejection_was_real
#axiomsClean standalone_tail_is_masked
#axiomsClean invalid_group_rejected
#axiomsClean scalar_numeric_not_automatic
#axiomsClean isa_numeric_not_automatic
#axiomsClean native_consistency
end VeriTile.Bench.Repairs.MatmulDequantInt4.Witnesses
