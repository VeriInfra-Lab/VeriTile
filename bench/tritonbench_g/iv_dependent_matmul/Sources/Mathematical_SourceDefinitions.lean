import bench.tritonbench_g.iv_dependent_matmul.Sources.Selected
import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
inductive Graph where | source | accumulator deriving DecidableEq, Repr
structure Parameters where
  (input weight output : RegionName)
  (rows columns inner inputRowStride inputInnerStride weightInnerStride weightColumnStride
    outputRowStride outputColumnStride blockRows blockColumns blockInner : Nat)
/-- Derived common view used by the execution decomposition, never a
standalone source authority. The complete repaired preLoad quotation supplies
the body; only its two initial pointer aliases are split out here. -/
def commonView (k : ComputeKernel) : ComputeKernel :=
  .mk k.inputs k.outputs (k.surfaceBody.take 9 ++ k.surfaceBody.drop 11)

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

def prologue (p : Parameters) := (source .source p).surfaceBody.take 10
def body (g : Graph) (p : Parameters) : List ComputeStmt :=
  let code : List ComputeStmt := (source g p).surfaceBody
  match code[10]? with | some (ComputeStmt.forRangeDyn _ _ _ _ b) => b | _ => []
def suffix (p : Parameters) := (source .source p).surfaceBody.drop 11

def stopOp (p : Parameters) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat p.inner) (.constNat p.blockInner))
    (.constNat 1)) (.constNat p.blockInner)
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
