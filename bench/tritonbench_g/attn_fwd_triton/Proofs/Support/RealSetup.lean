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
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealIteration
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealLoads
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealMemory
/-! Retained mathematical support for attn_fwd_triton. This module
provides no alternative kernel or finite-penalty source specification. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealSetup
open VeriTile.Triton RealLoads RealMemory RealIteration
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false

def lowerName (second : Bool) : RegName := if second then "second_lo" else "first_lo"

def upperName (second : Bool) : RegName := if second then "second_hi" else "first_hi"

def keyBase (second : Bool) : RegName := if second then "K_ptrs" else "K_ptrs_first"

def scaleBase (second : Bool) : RegName := if second then "K_scale_ptr" else "K_scale_ptr_first"

def valueBase (second : Bool) : RegName := if second then "V_ptrs" else "V_ptrs_first"

def lowerExpr (BM : Nat) : Op .nat [] :=
  .mul .nat .nil (.ref .nat [] "start_m") (.constNat BM)

def upperExpr (BM : Nat) : Op .nat [] :=
  .mul .nat .nil (.add .nat .nil (.ref .nat [] "start_m") (.constNat 1)) (.constNat BM)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealSetup