import bench.tritonbench_g.matmul_persistent_triton.Proofs.PairIO

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

abbrev original (P : Source.Arithmetic α α) (p : Core.Parameters) (sms : 0<p.problem.SMS) :=
  ProgramIO.Implementation.original (PublicData.io P p) (fun _ => True)
abbrev reordered (P : Source.Arithmetic α α) (p : Core.Parameters) (sms : 0<p.problem.SMS) :=
  ProgramIO.Implementation.ofVariant (PairIO.variant P p) (fun _ => True)

/-- The selected nested-loop scalar-cast original and its related row-index
candidate execute every actual persistent PID and assigned tile safely and
successfully under the original IO input binding and bounds, with equal complete
typed output cells and both exterior frames. Both FP16 and E4M3 storage
profiles retain their FP32 live accumulator, ordered dots and output conversion.
The rewrite is exact signed integer multiplication; floating primitives are
unchanged and require no strict or experimental numerical rewrite rule. -/
specification source_pair_equiv (P : Source.Arithmetic α α) (p : Core.Parameters)
    (sms : 0<p.problem.SMS) : original P p sms ≡[[]] reordered P p sms := by
  exact ProgramIO.Implementation.of_equivalent (PairIO.variant P p)
    (PairIO.equivalent P p sms)

#print_fp_assumptions source_pair_equiv
#axiomsClean source_pair_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.FPEquiv
