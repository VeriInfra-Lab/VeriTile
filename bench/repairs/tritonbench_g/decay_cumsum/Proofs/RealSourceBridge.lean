import bench.repairs.tritonbench_g.decay_cumsum.Proofs.RealBodyBridge
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Source
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.LegacySurface
import bench.tritonbench_g.decay_cumsum.Proofs.ForwardProjection
import bench.tritonbench_g.decay_cumsum.Proofs.PrepareProjection
import bench.tritonbench_g.decay_cumsum.Proofs.BackwardProjection

namespace VeriTile.Bench.Repairs.DecayCumsum.RealSourceBridge
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DecayCumsum
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def prefixCount : Kind → Nat | .forward => 7 | .prepare => 10 | .backward => 14
def prelude (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : List Stmt :=
  (Kernels.source kind h p scale).eraseDType.toAlgKernel.body.take (prefixCount kind)

def body (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : List Stmt :=
  match kind with
  | .forward => RealReference.fwdBody "g" "go" h.headStride p.span h.BK h.width
  | .prepare => RealReference.prepBody "q" "k" "g" "qg" "kg" h.headStride h.width p.span h.BK scale
  | .backward => RealBodyBridge.sourceBody h.width p.span h.BK

def loop (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : Stmt :=
  match kind with
  | .forward | .prepare => .forRangeInt "i" (.constInt 0)
      (Op.constNat p.span).castNatToInt (.constNat 1) (body kind h p scale)
  | .backward => .forRangeInt "__rev_j" (.sub .int .nil (.constInt 0) (.constInt ((p.span : Int)-1)))
      (.sub .int .nil (.constInt 0) (.sub .int .nil (.constInt 0) (.constInt 1)))
      (.constNat 1) (body kind h p scale)

theorem source_body (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) :
    (Kernels.source kind h p scale).eraseDType.toAlgKernel.body =
      prelude kind h p scale ++ [loop kind h p scale] := by
  cases kind
  all_goals simp only [Kernels.source,Kernels.quotedForward,Kernels.quotedPrepare,Kernels.quotedBackward,
    prelude,prefixCount,body,loop,RealReference.fwdBody,RealReference.prepBody,
    RealBodyBridge.sourceBody,RealReference.bwdIterBodyG,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,List.map_cons,List.map_nil,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,
    MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
    Except.bind,bind,eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType]
  all_goals rfl

end
end VeriTile.Bench.Repairs.DecayCumsum.RealSourceBridge
