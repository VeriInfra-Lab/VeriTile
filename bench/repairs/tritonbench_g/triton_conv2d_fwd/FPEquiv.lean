import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Pair

namespace VeriTile.Bench.Repairs.Conv2d.FPEquiv
open VeriTile Triton FP.Structural KernelNumeric
open scoped VeriTile.Spec
noncomputable section

def original (o : Operations A) (s : Shape) (f : Flags) :=
  ProgramIO.Implementation.original (ConvIO.program o s f) (fun _ => True)

def shifted (o : Operations A) (s : Shape) (f : Flags) :=
  ProgramIO.Implementation.ofVariant (Pair.variant o s f) (fun _ => True)

/-- Complete grouped-convolution source pair, retaining both operand-precision
paths, both TF32 settings, all h/w/c loops and the original full grid.
The sole change is exact fixed-128 row-base integer strength reduction. -/
specification grid_equiv {A : Type} [Inhabited A] (o : Operations A)
    (s : Shape) (valid : Valid s) (flags : Flags) :
    original o s flags ≡[[]] shifted o s flags :=
  ProgramIO.Implementation.of_equivalent (Pair.variant o s flags) (Pair.equivalent o s valid flags)

#axiomsClean grid_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Conv2d.FPEquiv
