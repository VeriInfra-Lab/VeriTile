import bench.tritonbench_g.matmul_triton2.Proofs.Pair

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.FPEquiv
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

def original (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) :=
  ProgramIO.Implementation.original (IO.matmul rest P p) (fun _ => True)
def maskHoisted (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) :=
  ProgramIO.Implementation.ofVariant (Pair.variant rest P p) (fun _ => True)

/-- Hoist invariant row/column comparisons out of the K loop, and reuse
both masks in every load and the final store. Every original configuration,
grouped grid, padded live TF32 dot and input domain is retained. Both full
schedules are safe and succeed, all outputs agree and both exact raw frames
hold. No floating rewrite rule is required. -/
specification source_pair_equiv (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (geometry : Geometry.Valid (Core.geometry p)) (inner : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    original rest P p ≡[[]] maskHoisted rest P p :=
  ProgramIO.Implementation.of_equivalent (Pair.variant rest P p)
    (Pair.equivalent rest P p geometry inner ha hb)

#print_fp_assumptions source_pair_equiv
#axiomsClean source_pair_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.FPEquiv
