import bench.tritonbench_g.chunk_cumsum_vector.Proofs.PublicNativeIO
import bench.tritonbench_g.chunk_cumsum_vector.IO
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FPEquiv
open VeriTile.Triton FP.Structural FPValues
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

/-- Compatibility names for the displayed actual IO and independent domain. -/
abbrev prefixNativeIO := @FloatingIO.prefixNativeIO
abbrev sumLeftVariant := @FloatingIO.sumLeftVariant
abbrev carryDomain := @FloatingIO.carryDomain
abbrev original := @FloatingIO.original
abbrev sumLeft := @FloatingIO.sumLeft

/-- Both complete source invocations succeed, have bounded actual accesses,
agree on every active typed output and preserve their exact exterior frames.
The logical carry domain is proved equivalent to the prior reached-site domain.
The carry swap uses the fixed strict FP32 theory with no empirical rules. -/
specification memory_equiv (M : Algebra α) (dot : DotBackend α)
    (D : FP.Guarded.Domain α) (model : FP.Guarded.Models [] M D)
    (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (chosen : c∈configurations)
    (p : Geometry.Program h) :
    original M dot D h c p ≡[[]] sumLeft M dot D h c p :=
  ProgramIO.Implementation.of_equivalentOn (sumLeftVariant M dot h c p)
    (NativeIO.equivalent M dot D model h layout c chosen p)

#print_fp_assumptions memory_equiv
#specNonCircular carryDomain avoiding [Kernels.source,Kernels.sumLeft,run,FPExecution.source_correct]
#axiomsClean memory_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.FPEquiv
