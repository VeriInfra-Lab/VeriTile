import bench.repairs.tritonbench_g.masked_select.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.MaskedSelect.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MaskedSelect TypedCopy Whole
open scoped VeriTile.Spec
noncomputable section

/-- Two actual complete autotune configurations, retaining exact payload,
Bool and prefix input cells. Both schedules succeed safely, agree on active
outputs and preserve the entire exterior frame on the original prefix domain. -/
specification masked_select_configuration_equiv {α : Type} [Inhabited α]
    (rest : Algebra α) (p : Parameters) (left right : Config)
    (hl : left ∈ configs) (hr : right ∈ configs) (separate : KernelSeparate p) :
    IO.original rest p left ≡[[]] IO.configured rest p left right :=
  ProgramIO.Implementation.of_equivalentOnCells (IO.candidate rest p left right)
    (PublicIO.equivalence rest p left right hl hr separate)

#print_fp_assumptions masked_select_configuration_equiv
#axiomsClean masked_select_configuration_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedSelect.FPEquiv
