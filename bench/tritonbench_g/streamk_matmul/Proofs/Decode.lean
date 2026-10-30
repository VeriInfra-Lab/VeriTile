import bench.tritonbench_g.streamk_matmul.Execution
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
open Whole001 Source
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

/-- The executable plan is obtained from the quotation, not maintained as a
second handwritten AST. This witness is only source decoding, not execution. -/
def sourcePlan : Program := (compile kernels).get (by decide)

theorem source_compiles : compile kernels = some sourcePlan := by rfl
theorem source_typed : typed sourcePlan = Bool.true := by decide

abbrev mlane (program : Nat) (lane : Lane p) (k : Nat) : Lanes :=
  ⟨program,lane.1.val,lane.2.val,k⟩

theorem mac_row (p : Source.Parameters) (program : Nat) (c : Protocol.Chunk)
    (lane : Lane p) (k : Nat) :
    tileEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane k) "rm" =
      Source.row p (c.tile (Plan.select p.h).iterations) lane := by rfl

theorem mac_col (p : Source.Parameters) (program : Nat) (c : Protocol.Chunk)
    (lane : Lane p) (k : Nat) :
    tileEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane k) "rn" =
      Source.col p (c.tile (Plan.select p.h).iterations) lane := by rfl

theorem mac_maskA (p : Source.Parameters) (program : Nat) (c : Protocol.Chunk)
    (lane : Lane p) (k iter : Nat) :
    sourcePlan.mac.core.loop.maskA.eval
      (loopEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane k) iter)
      (mlane program lane k) ↔
    Source.row p (c.tile (Plan.select p.h).iterations) lane<p.h.m ∧
      (iter%(Plan.select p.h).iterations)*p.h.bk+k<p.h.k := by rfl

theorem mac_maskB (p : Source.Parameters) (program : Nat) (c : Protocol.Chunk)
    (lane : Lane p) (k iter : Nat) :
    sourcePlan.mac.core.loop.maskB.eval
      (loopEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane k) iter)
      (mlane program lane k) ↔
    (iter%(Plan.select p.h).iterations)*p.h.bk+k<p.h.k ∧
      Source.col p (c.tile (Plan.select p.h).iterations) lane<p.h.n := by rfl

#axiomsClean source_compiles
#axiomsClean source_typed
#axiomsClean mac_row
#axiomsClean mac_col
#axiomsClean mac_maskA
#axiomsClean mac_maskB
#specNonCircular sourcePlan avoiding [Source.call,Source.firstWave,Source.fullWave,Source.reduceWave]
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
