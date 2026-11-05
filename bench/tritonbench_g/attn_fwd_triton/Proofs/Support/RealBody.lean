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
/-! Retained mathematical support for attn_fwd_triton. This module
provides no alternative kernel or finite-penalty source specification. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealBody
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false

def keyMask (BD BN N HA : Nat) : Op .bool [BD,BN] :=
  .boolAnd (.consL (.consR .nil))
    (.lt .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [BN] "offs_n"))
      (.sub .nat .nil (.constNat N) (.ref .nat [] "start_n")))
    (.expandDim ⟨1, by simp⟩ (.lt .nat .scalarR (.arange BD) (.constNat HA)))

def scoreExpr (BM BN BD : Nat) : Op .real [BM,BN] :=
  .mul .real .scalarR
    (.mul .real .scalarR (.dot (batch := [])
      (.ref .real [BM,BD] "q") (.ref .real [BD,BN] "k")) (.ref .real [] "q_scale"))
    (.ref .real [] "k_scale")

def score (BM BN BD : Nat) : Stmt := .assign .real [BM,BN] "qk" (scoreExpr BM BN BD)

def centerExpr (BM BN : Nat) : Op .real [BM] :=
  .max2 (.consSame .nil) (.ref .real [BM] "m_i")
    (.reduceMax ⟨1, by simp⟩ Bool.false (.ref .real [BM,BN] "qk"))

def center (BM BN : Nat) : List Stmt :=
  [.assign .real [BM] "m_ij" (centerExpr BM BN),
   .assign .real [BM,BN] "qk" (.sub .real (.consSame (.consR .nil))
     (.ref .real [BM,BN] "qk") (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "m_ij")))]

def probability (BM BN : Nat) : Stmt :=
  .assign .real [BM,BN] "p" (.exp2 (.ref .real [BM,BN] "qk"))

def rescale (BM BN BD : Nat) : List Stmt :=
  [.assign .real [BM] "l_ij" (.reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [BM,BN] "p")),
   .assign .real [BM] "alpha" (.exp2 (.sub .real (.consSame .nil)
     (.ref .real [BM] "m_i") (.ref .real [BM] "m_ij"))),
   .assign .real [BM] "l_i" (.add .real (.consSame .nil)
     (.mul .real (.consSame .nil) (.ref .real [BM] "l_i") (.ref .real [BM] "alpha"))
     (.ref .real [BM] "l_ij")),
   .assign .real [BM,BD] "acc" (.mul .real (.consSame (.consR .nil))
     (.ref .real [BM,BD] "acc") (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "alpha")))]

def accumulate (BM BN BD : Nat) : List Stmt :=
  [.assign .real [BM,BN] "p" (.ref .real [BM,BN] "p"),
   .assign .real [BM,BD] "acc" (.add .real (.consSame (.consSame .nil))
     (.ref .real [BM,BD] "acc") (.dot (options := {outDType := some .fp16}) (batch := [])
       (.ref .real [BM,BN] "p") (.ref .real [BN,BD] "v"))),
   .assign .real [BM] "m_i" (.ref .real [BM] "m_ij")]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealBody