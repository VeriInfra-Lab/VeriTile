import bench.tritonbench_g.chunk_gla_fwd.Sources.Mathematical_RealOutput
import bench.tritonbench_g.chunk_gla_fwd.Sources.Mathematical_RealOutputSource
import VeriTile.Triton.Math.MatrixRewrite

/-! Transfer the complete real execution to every typed storage choice and
the candidate. Both source decompositions are checked by Lean; the only
semantic change is the order of a real matrix addition. This module does not
use FP atoms and is not imported by the floating-point proof. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputCandidate
open VeriTile.Triton RealOutput
set_option maxRecDepth 20000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
section

variable (query value gate : TypedOutput.Storage)
    (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat)





def intraDot (BT BV : Nat) : Op .real [BT,BV] :=
  .dot (batch := []) (.ref .real [BT,BT] "b_A") (.ref .real [BT,BV] "b_v")
    {allowTF32 := some Bool.false}








end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputCandidate
