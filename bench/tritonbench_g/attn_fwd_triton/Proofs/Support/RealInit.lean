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
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealPointers
/-! Retained mathematical support for attn_fwd_triton. This module
provides no alternative kernel or finite-penalty source specification. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealInit
open VeriTile.Triton RealLoads RealPointers
open AttentionFwdTriton3.RealTiles (finite)
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def queryMask (BM BD N HA : Nat) : Op .bool [BM,BD] :=
  .boolAnd (.consR (.consL .nil))
    (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [BM] "offs_m")) (.constNat N))
    (.expandDim ⟨0, by simp⟩ (.lt .nat .scalarR (.arange BD) (.constNat HA)))

def code (BM BD N HA : Nat) : List Stmt :=
  [.assign .real [BM] "m_i" (.add .real .scalarR (.full [BM] (.const 0)) .negInf),
   .assign .real [BM] "l_i" (.add .real .scalarR (.full [BM] (.const 0)) (.const 1.0)),
   .assign .real [BM,BD] "acc" (.full [BM,BD] (.const 0)),
   .assign .real [BM,BD] "q" (.load .real (.ptr (.ref .ptr [BM,BD] "Q_ptrs")) (.mask (queryMask BM BD N HA))),
   .assign .real [] "q_scale" (.load .real (.ptr (.ref .ptr [] "Q_scale_ptr")) .none)]

def mask (BM BD startM N HA : Nat) : Tile .bool [BM,BD] :=
  ⟨fun i => decide (startM*BM+i.1.val < N ∧ i.2.1.val < HA)⟩

def writes : List RegName := ["m_i","l_i","acc","q","q_scale"]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealInit