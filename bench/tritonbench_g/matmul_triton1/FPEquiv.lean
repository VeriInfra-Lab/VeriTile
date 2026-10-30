import bench.tritonbench_g.matmul_triton1.Proofs.Pair

/-! This entry proves the selected compiler graph. The separate
`pythonBody` / `pythonSource` constructors in Kernels.lean retain the original
standalone dot followed by addition; no source-to-compiler FP equivalence is
asserted by this result. -/

namespace VeriTile.Bench.Repairs.MatmulTriton1.FPEquiv
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

def original (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) :=
  ProgramIO.Implementation.original (MatmulIO.io rest P p) (fun _ => True)
def maskHoisted (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) :=
  ProgramIO.Implementation.ofVariant (Pair.variant rest P p) (fun _ => True)

/-- Cache the loop-invariant row/column comparisons once per program and
reuse them in every K-loop load and the final store. Both complete source
grids are safe and succeed under the retained input/stride/finite-index
domain; every output cell agrees and both exact exterior frames hold.
The FP32 live ordered IEEE dot graph is unchanged. -/
specification source_pair_equiv (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (canonical : FiniteSource.Canonical p) (ready : FiniteIndex.Ready p)
    (storage : FiniteIndex.StorageBounds p) (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    original rest P p ≡[[]] maskHoisted rest P p :=
  ProgramIO.Implementation.of_equivalent (Pair.variant rest P p)
    (Pair.equivalent rest P p canonical ready storage geometry ha hb)

#print_fp_assumptions source_pair_equiv
#axiomsClean source_pair_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulTriton1.FPEquiv
