import bench.tritonbench_g.context_attn_fwd.Proofs.Sources.Mathematical_Program
import VeriTile.Triton.Semantics.BroadcastReshape
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.Prelude
open VeriTile.Triton Formula Program
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def prefixSource (p : Layout) (r : Regions) : ComputeKernel :=
  .mk [] [] (parts p r Bool.false 0).before

theorem prefix_source (p : Layout) (r : Regions) :
    prelude p r = (prefixSource p r).eraseDType.toAlgKernel.body := by
  simp only [prelude,prefixSource,lower,parts,mathematical,kernel,VeriTile.Bench.TritonBenchG.ContextAttnFwd.original,VeriTile.Bench.Repairs.ContextAttnFwd.SelectedSource.original,VeriTile.Bench.Repairs.ContextAttnFwd.originalKernel,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,Except.toOption,Option.getD,
    Kernel.eraseDType,Stmt.eraseDTypeList,List.map_cons,List.map_nil,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,VeriTile.Triton.eraseDType,
    NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    Source.intRange?,String.reduceEq,Bool.false_eq_true,↓reduceIte,Option.map_some]

end VeriTile.Bench.TritonBenchG.ContextAttnFwd.Prelude
