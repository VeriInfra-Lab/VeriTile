import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativePublic

/-! Retained single-source native correctness. These ordinary theorems are
auxiliary results; their historical namespaces are preserved for callers. -/

namespace VeriTile.Bench.Repairs.ContextAttnFwd.PublicKernel
open VeriTile.Triton FP.Structural NativeGraph
noncomputable section

/-- Original FP32 source under its fixed SM90 native primitive model.
Every finite launch list, raw typed-array alias and native consumer is retained.
This is source-to-native observation; no arithmetic rewrite or Real-equality
law is assumed, and there is no second optimization kernel. -/
theorem correct_fp32 (p : Layout) (r : Regions) (scale : ℝ)
    (jobs : List NativeFormula.Launch) (positive : 0 < p.BN) (ops : ScalarOps) (owner : Owner) :
    (NativePublic.io p r scale jobs).Observes (NativePublic.observer p r scale jobs ops owner)
      (NativePublic.formula p r scale jobs ops owner) :=
  NativePublic.complete p r scale jobs positive ops owner

#stmtSurfaceSubset correct_fp32 ⊆ [NativePublic.io,NativePublic.observer,NativePublic.formula,
  ArrayProgramIO.Observes,Layout,Regions,NativeFormula.Launch,ScalarOps,Owner,
  NativeGraph.Expr,NativeGraph.Word,NativeGraph.instInhabitedExpr,
  VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.Layout.BN]
#axiomsClean correct_fp32
end
end VeriTile.Bench.Repairs.ContextAttnFwd.PublicKernel
