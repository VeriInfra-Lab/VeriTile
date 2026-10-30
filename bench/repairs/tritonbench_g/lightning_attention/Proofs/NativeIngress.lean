import bench.repairs.tritonbench_g.lightning_attention.Proofs.NativeComposition
import bench.repairs.tritonbench_g.lightning_attention.Proofs.HostAPI
namespace VeriTile.Bench.Repairs.LightningAttention.Native
open VeriTile Triton FP.Structural
noncomputable section
/-- IEEE FP32 unary negation flips the sign bit, including signed zero and NaNs. -/
def negateWord (w : Word) : Word := w ^^^ 0x80000000
/-- One local library primitive: PyTorch resolve_neg applied to one FP32 word.
This is explicitly deferred, separately from ordinary copy/layout bindings. -/
structure NegationConformance (actual : Word → Word) : Prop where
  negate : ∀ w, actual w=negateWord w

def logicalWords (a : Host.Allocations) (physical : Nat → Cell Word) (fallback : Memory Word) : WordMemory :=
  fun r o => (Host.logical a physical fallback r o).read .real

theorem forward_copies (actual : Word → Word) (numeric : NegationConformance actual)
    (c : Host.Invocation) (a : Host.Allocations) (before after : Nat → Cell Word)
    (copies : Host.ForwardCopies actual c a before after) : Host.ForwardCopies negateWord c a before after := by
  have he : actual=negateWord := funext numeric.negate
  simpa only [he] using copies

theorem incoming_copy (actual : Word → Word) (numeric : NegationConformance actual)
    (c : Host.Invocation) (a : Host.Allocations) (before after : Nat → Cell Word)
    (copies : Host.IncomingCopy actual c a before after) : Host.IncomingCopy negateWord c a before after := by
  have he : actual=negateWord := funext numeric.negate
  simpa only [he] using copies
#axiomsClean forward_copies
#axiomsClean incoming_copy
#specNonCircular NegationConformance avoiding [stageExecute,gridExecute,execute,reference]
end
end VeriTile.Bench.Repairs.LightningAttention.Native
