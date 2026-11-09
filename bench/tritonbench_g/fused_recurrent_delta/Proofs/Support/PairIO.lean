import bench.tritonbench_g.fused_recurrent_delta.IO
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelNativeCarrier
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! Actual runtime program views over the one shared memory interface. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.IOPair
open VeriTile.Triton FP.Structural KernelProgram Interface
noncomputable section
variable {α : Type} [Inhabited α]

 def variant (M : Algebra α) (kind : Kind) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) :
    ProgramIO.Variant (RuntimeIO.io M kind Bool.false s f scale bh iv) where
  schedule := fun _ => [RuntimeIO.invocation M kind Bool.true s f scale bh iv]
  observe := fun _ _ => id

 def original (isa : KernelNativeCarrier.Instructions) (routing : KernelNativeCarrier.Routing)
    (scale : KernelNativeCarrier.Word) (kind : Kind) (s : Shape) (f : Flags) (bh iv : Nat) :=
  ProgramIO.Implementation.original
    (RuntimeIO.io (KernelNativeCarrier.algebra isa routing scale) kind Bool.false s f KernelNativeCarrier.scaleSlot bh iv)
    (fun _ => True)
 def candidate (isa : KernelNativeCarrier.Instructions) (routing : KernelNativeCarrier.Routing)
    (scale : KernelNativeCarrier.Word) (kind : Kind) (s : Shape) (f : Flags) (bh iv : Nat) :=
  ProgramIO.Implementation.ofVariant
    (variant (KernelNativeCarrier.algebra isa routing scale) kind s f KernelNativeCarrier.scaleSlot bh iv)
    (fun _ => True)

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.IOPair
