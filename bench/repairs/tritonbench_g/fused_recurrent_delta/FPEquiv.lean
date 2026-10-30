import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.IOPair
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.FPEquiv
open VeriTile.Triton FP.Structural KernelProgram IOPair
open scoped VeriTile.Spec
noncomputable section

/-- Complete actual runtime program pair, for both stages. Forward removes
the zero key-grid output-address term; backward transparently reuses source.
All native words and the original scalar/vector beta layouts are retained. -/
specification kernel_equivalent (isa : KernelNativeCarrier.Instructions)
    (routing : KernelNativeCarrier.Routing) (scale : KernelNativeCarrier.Word)
    (kind : Kind) (s : Shape) (f : Flags) (bh iv : Nat) :
    original isa routing scale kind s f bh iv ≡[[]] candidate isa routing scale kind s f bh iv := by
  apply ProgramIO.Implementation.of_equivalent
  exact IOPair.complete (KernelNativeCarrier.algebra isa routing scale) kind s f KernelNativeCarrier.scaleSlot bh iv

#axiomsClean kernel_equivalent
#print_fp_assumptions kernel_equivalent
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.FPEquiv
