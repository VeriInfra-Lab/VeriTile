import VeriTile.Torch.Tensor
import bench.tritonbench_g.vector_addition.Proofs.NativeArithmetic

/-! The original wrapper's assertions do not establish its kernel input
contract. These counterexamples do not change admission or the source. -/
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.SourceDefect
open VeriTile.Torch

def encoderAssertions (x : TensorMetadata) : Prop :=
  (x.dtype = .other "float16" ∨ x.dtype = .floating .fp32) ∧ x.device.kind = "cuda"

def slice : TensorMetadata :=
  ⟨.floating .fp32, .strided, ⟨"cuda", 0⟩, [3], Bool.true, Bool.false, 4104⟩

/-- x = backing[2:5], backing has eight FP32 elements. The wrapper passes
storage_size / element_size = 8, whereas empty_like owns three output bytes.
Lane three is launched and active, and writes the first out-of-bounds byte. -/
theorem admitted_slice_overrun :
    encoderAssertions slice ∧ slice.numel = 3 ∧
    (0 < (slice.numel + 1023) / 1024) ∧ (3 < 1024) ∧ (3 < 32 / 4) ∧
    ¬ (3 < slice.numel) := by
  unfold encoderAssertions
  decide +kernel

def strided : TensorMetadata :=
  ⟨.floating .fp32, .strided, ⟨"cuda", 0⟩, [3], Bool.false, Bool.false, 4096⟩

/-- x = backing[0:6:2]. Logical coordinate one lives at element two;
the source's flat pointer instead loads element one. Admission allows it. -/
theorem admitted_stride_mismatch :
    encoderAssertions strided ∧ strided.isContiguous = Bool.false ∧
      (4096 + 1*4 : Nat) ≠ 4096 + (1*2)*4 := by
  unfold encoderAssertions
  decide +kernel

/-- Even when N is promoted beyond i32 by the launch ABI, the source's
pid*1024 offset is still signed i32 and sign-extends to a negative address.
This theorem concerns that expression only, without guessing N's ABI. -/
theorem signed_offset_overflow :
    (VectorAddition.Native.offset (2^21) 0).toInt = -(2^31 : Int) ∧
    VectorAddition.Native.pointer (2^33+32) 4 (2^21) 0 = 32 :=
  ⟨VectorAddition.Native.wrapped_negative_lane.1,
    VectorAddition.Native.wrapped_pointer_can_reenter⟩

#axiomsClean admitted_slice_overrun
#axiomsClean admitted_stride_mismatch
#axiomsClean signed_offset_overflow
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.SourceDefect
