import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics.EvalOp
import VeriTile.Triton.Source
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealAccess
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealFormula
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealKeys
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealLoopFormula
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealProbabilities
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealRegisterSafety
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealTiles
import bench.tritonbench_g.attention_score.Proofs.Mathematical_RealFrame
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealLoads
/-! Retained mathematical support for attn_fwd_triton. This module
provides no alternative kernel or finite-penalty source specification. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealPointers
open VeriTile.Triton RealLoads
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def rowExpr (BM : Nat) : Op .nat [BM] :=
  .add .nat .scalarL (.mul .nat .nil (.ref .nat [] "start_m") (.constNat BM)) (.arange BM)

def queryScaleExpr (QS : RegionName) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase QS) (.add .nat .nil (.ref .nat [] "q_scale_offset") (.ref .nat [] "start_m"))

def keyScaleExpr (KS : RegionName) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase KS) (.ref .nat [] "k_scale_offset")

def rows (BR start : Nat) : Tile .nat [BR] := ⟨fun i => start+i.1.val⟩

def rowPointer (R : RegionName) (BR BD base start strideRow strideDim : Nat) : Tile .ptr [BR,BD] :=
  ⟨fun i => (R.cast,base+(start+i.1.val)*strideRow+i.2.1.val*strideDim)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealPointers