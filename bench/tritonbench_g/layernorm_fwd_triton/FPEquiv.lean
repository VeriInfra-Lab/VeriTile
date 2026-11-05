import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Support.PairIO
import VeriTile.Triton.Float.ClosedStructuralAudit

/-! The actual complete candidate schedules each output weight load after
x_hat, shortening the weight vector's lifetime. The three-pass reduction order,
FP32 arithmetic, arbitrary strided input scope and full sequence/head grid stay.
The compiler and native instruction observations remain in Proofs/. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton.FPEquiv
open VeriTile.Triton FP.Structural Address
open scoped VeriTile.Spec
noncomputable section

abbrev original := @PairIO.original
abbrev candidate := @PairIO.candidate

/-- Exact complete-grid scheduling equivalence. The existing output-coordinate
injectivity condition retains precisely the original arbitrary-stride scope. -/
specification source_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (shape : Shape) (st : Strides) (eps : α)
    (injective : Function.Injective (fun q : Coordinate shape => yOffset st q.1 q.2.val)) :
    original M shape st eps ≡[[]] candidate M shape st eps :=
  ProgramIO.Implementation.of_equivalent (PairIO.variant M shape st eps)
    (PairIO.equivalent M shape st eps injective)

-- Audit every reachable project declaration, including unused branches;
-- reject numerical constructors, opaque code and external FP certificates.
#print_fp_assumptions source_equivalence closed_structural
#axiomsClean source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.FPEquiv
