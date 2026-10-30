import VeriTile.Triton.Float.ScalarOps
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Native

namespace VeriTile.Bench.Repairs.Conv2d.Software
open VeriTile.Triton Native

def pack (w : Word) : BitVec 32 := w 3 ++ w 2 ++ w 1 ++ w 0
def unpack (bits : BitVec 32) : Word := fun i=>bits.extractLsb' (8*i.val) 8
def value (w : Word) : FP.Value .fp32 := ⟨pack w⟩
def word (x : FP.Value .fp32) : Word := unpack x.bits

/-- Default binary32 numerical reference: one correctly rounded fused
multiply-add, round-to-nearest-even, preserving subnormal inputs/results.
Special values follow the explicit software profile's canonical NaN rule. -/
def fmaRN (a b carry : Word) : Word :=
  word (FP.fma .ieeeValue (value a) (value b) (value carry))

/-- Non-default numerical ports have literal, fixed ISA signatures. These
represent exactly one conversion or one SM90 WGMMA, never a configurable dot. -/
structure TensorNumerics where
  halfRN : Word → Half
  wgmma16 : Mat 64 16 Half → Mat 16 32 Half → Frag Word → Frag Word
  wgmma8 : Mat 64 8 Word → Mat 8 32 Word → Frag Word → Frag Word

def instructions (tensor : TensorNumerics) : Instructions :=
  ⟨fmaRN,tensor.halfRN,tensor.wgmma16,tensor.wgmma8⟩

def reference (tensor : TensorNumerics) (s : Shape) (flags : Flags) (m : Memory Word) (x : Cell) :=
  ordered s (isa (instructions tensor)) flags m x

/-- These unused fields make the default reference closed: the false/false
branch never invokes a conversion or tensor instruction. -/
def unusedTensor : TensorNumerics :=
  ⟨fun _ _=>0,fun _ _ carry=>carry,fun _ _ carry=>carry⟩
def defaultReference (s : Shape) (m : Memory Word) (x : Cell) :=
  reference unusedTensor s ⟨Bool.false,Bool.false⟩ m x

theorem default_reference (tensor : TensorNumerics) (s : Shape) (m : Memory Word) (x : Cell) :
    reference tensor s ⟨Bool.false,Bool.false⟩ m x=defaultReference s m x := by rfl

#axiomsClean default_reference
#specNonCircular reference avoiding [source,sourceValue,sourceProgram,sourcePrograms,Native.selected]
#specNonCircular fmaRN avoiding [source,sourceValue,sourceProgram,sourcePrograms,Native.selected]
end VeriTile.Bench.Repairs.Conv2d.Software
