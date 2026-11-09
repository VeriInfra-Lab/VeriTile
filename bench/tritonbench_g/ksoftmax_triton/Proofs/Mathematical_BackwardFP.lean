import bench.tritonbench_g.ksoftmax_triton.Proofs.Sources.Kernels
import VeriTile.Triton.Float.IndexExpressionRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Floating
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

@[spec_rule] theorem index_product_commutes {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (a b : Op .nat []) :
    OpEquivalent M s before (.mul .nat .nil a b) (.mul .nat .nil b a) :=
  FP.ProfiledRewrite.op_nat_mul_comm M s before a b

@[spec_rule] theorem pointer_add_congr {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (bc : Broadcast sa sb sh)
    (p p' : Op .ptr sa) (a a' : Op .nat sb)
    (hp : OpEquivalent M s before p p') (ha : OpEquivalent M s before a a') :
    OpEquivalent M s before (.ptrAdd bc p a) (.ptrAdd bc p' a') :=
  FP.ProfiledRewrite.op_ptr_add_congr M s before bc p p' a a' hp ha

def assumptions : Spec.Assumptions FP.GuardedFragment := []

#axiomsClean index_product_commutes
#axiomsClean pointer_add_congr
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Floating
