import VeriTile.Triton.Float.IndexExpressionRewrite
import bench.tritonbench_g.cache_transform_triton.Proofs.Sources.Mathematical_IndexRewrite
namespace VeriTile.Bench.TritonBenchG.DestindexCopy.IndexRewrite
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
/-- Equal address expressions preserve pointer evaluation, including failure. -/
@[spec_rule] theorem op_pointer_add {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (bc : Broadcast sa sb sh)
    (p p' : Op .ptr sa) (a a' : Op .nat sb)
    (hp : OpEquivalent M s before p p') (ha : OpEquivalent M s before a a') :
    OpEquivalent M s before (.ptrAdd bc p a) (.ptrAdd bc p' a') :=
  FP.ProfiledRewrite.op_ptr_add_congr M s before bc p p' a a' hp ha
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DestindexCopy.IndexRewrite
