import bench.tritonbench_g.fast_rms_layernorm.Proofs.RuntimeForward
import bench.tritonbench_g.fast_rms_layernorm.Proofs.EpsilonABI
import bench.tritonbench_g.fast_rms_layernorm.Proofs.NativeGraph

/-! Input byte interpretation and zero encoding used by the native kernel proofs.
The namespace is retained so their public premises and reference terms do not change. -/
namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPEquiv
open VeriTile Triton FP.Structural KernelTensor KernelStorage NativeGraph
noncomputable section
open scoped BigOperators

/-- Little-endian FP32 storage words; every exceptional encoding is admitted. -/
def wordOfBytes (bytes : Fin 4 → Fin 256) : Word :=
  BitVec.ofNat 32 (∑ byte : Fin 4,(bytes byte).val * 2^(8*byte.val))
def decodeBytes (bytes : Fin 4 → Fin 256) : Cell Term := .mk .real (.bits (wordOfBytes bytes))
def logicalTerms (logical : RegionName → Nat → Word) : RegionName → Nat → Term :=
  fun role off => .bits (logical role off)
def WordReadLaw (phase : Phase) (tensors : RegionName → Torch.TensorMetadata)
    (heap : Nat → Fin 256) (logical : RegionName → Nat → Word) : Prop :=
  ∀ role,role∈inputs phase → ∀ off,off<(tensors role).numel →
    wordOfBytes (fun byte => heap ((tensors role).dataPtr.toNat+off*4+byte.val))=logical role off

theorem word_reads (phase : Phase) (tensors : RegionName → Torch.TensorMetadata)
    (heap : Nat → Fin 256) (logical : RegionName → Nat → Word) (reads : WordReadLaw phase tensors heap logical) :
    ReadLaw phase tensors heap decodeBytes (Cell.read .real) (logicalTerms logical) := by
  intro role hr off hi
  exact congrArg Term.bits (reads role hr off hi)

def zero : Cell Term := .mk .real (.bits 0)
theorem zero_bytes (I : Instructions) (S : Selection) (byte : Fin 4) : encode I S zero byte=0 := by
  apply Fin.ext
  simp [encode,observe,zero,Cell.read,eval,lower,Expr.eval]

#axiomsClean word_reads
#axiomsClean zero_bytes
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPEquiv
