import bench.tritonbench_g.iv_dependent_matmul.Proofs.Sources.Selected
import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
/-- Derived common view used by the execution decomposition, never a
standalone source authority. The complete repaired preLoad quotation supplies
the body; only its two initial pointer aliases are split out here. -/
def commonView (k : ComputeKernel) : ComputeKernel :=
  .mk k.inputs k.outputs (k.surfaceBody.take 18 ++ k.surfaceBody.drop 20)

def typed (dtype : FloatDType)
    (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel :=
  commonView (VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadSource
    dtype A B C M N K sam sak sbk sbn scm scn BM BN BK)

def fusedTyped (dtype : FloatDType)
    (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK : Nat) : ComputeKernel :=
  commonView (VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadNative
    dtype A B C M N K sam sak sbk sbn scm scn BM BN BK)

def source (g : Graph) (p : Parameters) : ComputeKernel :=
  (match g with | .source => typed | .accumulator => fusedTyped) .fp32
    p.input p.weight p.output p.rows p.columns p.inner p.inputRowStride p.inputInnerStride
    p.weightInnerStride p.weightColumnStride p.outputRowStride p.outputColumnStride
    p.blockRows p.blockColumns p.blockInner

def prologue (p : Parameters) := (source .source p).surfaceBody.take 19
def body (g : Graph) (p : Parameters) : List ComputeStmt :=
  let code : List ComputeStmt := (source g p).surfaceBody
  match code[19]? with | some (ComputeStmt.forRangeInt _ _ _ _ b) => b | _ => []
def suffix (p : Parameters) := (source .source p).surfaceBody.drop 20

def stopOp (p : Parameters) : Op .int [] :=
  let code : List ComputeStmt := (source .source p).surfaceBody
  match code[19]? with
  | some (ComputeStmt.forRangeInt _ _ stop _ _) => stop
  | _ => .constInt 0
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
