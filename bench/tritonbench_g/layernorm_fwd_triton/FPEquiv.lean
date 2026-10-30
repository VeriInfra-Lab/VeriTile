import bench.tritonbench_g.layernorm_fwd_triton.PairIO

/-! The actual complete candidate schedules each output weight load after
x_hat, shortening the weight vector's lifetime. The three-pass reduction order,
FP32 arithmetic, arbitrary strided input scope and full sequence/head grid stay.
Optional compiler/native observations remain in Misc/NativeCorrect.lean. -/
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

#axiomsClean source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.FPEquiv
