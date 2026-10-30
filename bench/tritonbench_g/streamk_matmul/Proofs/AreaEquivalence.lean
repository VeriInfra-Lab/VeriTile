import bench.tritonbench_g.streamk_matmul.Proofs.EventOriginalDomain
import bench.tritonbench_g.streamk_matmul.Proofs.EventCapacity
import bench.tritonbench_g.streamk_matmul.ProgramView

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.AreaEquivalence
open Whole001 Source Reader VeriTile Triton
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 300000

def areaPlan : Reader.Program := (Reader.compile areaKernels).get (by decide)
theorem candidate_compiles : Reader.compile areaKernels = some areaPlan := rfl

theorem mac_slots (e : Reader.Env) (l : Reader.Lanes) :
    sourcePlan.mac.slots.eval e l = areaPlan.mac.slots.eval e l := by
  change (e "program" * e "BLOCK_M") * e "BLOCK_N" + _ + _ =
    e "program" * (e "BLOCK_M" * e "BLOCK_N") + _ + _
  rw [Nat.mul_assoc]

theorem reduce_slots (e : Reader.Env) (l : Reader.Lanes) :
    sourcePlan.reduce.slots.eval e l = areaPlan.reduce.slots.eval e l := by
  change (e "pid" * e "BLOCK_M") * e "BLOCK_N" + _ + _ =
    e "pid" * (e "BLOCK_M" * e "BLOCK_N") + _ + _
  rw [Nat.mul_assoc]

theorem mac_events (p : Parameters) (ops : Instructions α) (pid : Fin p.h.programs)
    (c : Protocol.Chunk) :
    Reader.macEvents p sourcePlan.mac ops pid c = Reader.macEvents p areaPlan.mac ops pid c := by
  change Reader.macEvents p sourcePlan.mac ops pid c = Reader.macEvents p
    {sourcePlan.mac with slots := areaPlan.mac.slots} ops pid c
  simp only [Reader.macEvents,Reader.macEnv,mac_slots]

theorem reduce_env (p : Parameters) (tile pid : Nat) (lane : Lane p) :
    Reader.reduceEnv p sourcePlan.reduce tile pid lane =
      Reader.reduceEnv p areaPlan.reduce tile pid lane := by
  change Reader.reduceEnv p sourcePlan.reduce tile pid lane = Reader.reduceEnv p
    {sourcePlan.reduce with slots := areaPlan.reduce.slots} tile pid lane
  simp only [Reader.reduceEnv,reduce_slots]

theorem reduce_events (p : Parameters) (tile : Nat) :
    Reader.reduceEvents (α:=α) p sourcePlan.reduce tile = Reader.reduceEvents p areaPlan.reduce tile := by
  simp only [Reader.reduceEvents,reduce_env]
  rfl

theorem first_events_congr (p : Parameters) (ops : Instructions α) (a z : Reader.Program)
    (first : a.first = z.first)
    (mac : ∀ pid c, Reader.macEvents p a.mac ops pid c = Reader.macEvents p z.mac ops pid c) :
    Reader.firstEvents p a ops = Reader.firstEvents p z ops := by
  have macFunction (pid : Fin p.h.programs) : Reader.macEvents p a.mac ops pid = Reader.macEvents p z.mac ops pid := funext (mac pid)
  simp only [Reader.firstEvents,first,macFunction]

theorem first_events (p : Parameters) (ops : Instructions α) :
    Reader.firstEvents p sourcePlan ops = Reader.firstEvents p areaPlan ops :=
  first_events_congr p ops sourcePlan areaPlan rfl (mac_events p ops)

theorem events_congr (p : Parameters) (ops : Instructions α) (a z : Reader.Program)
    (typedEqual : Reader.typed a = Reader.typed z)
    (first : Reader.firstEvents p a ops = Reader.firstEvents p z ops)
    (reduce : ∀ tile, Reader.reduceEvents (α:=α) p a.reduce tile = Reader.reduceEvents p z.reduce tile)
    (blocking : a.full = z.full) : Reader.events p a ops = Reader.events p z ops := by
  have reduceFunction : Reader.reduceEvents (α:=α) p a.reduce = Reader.reduceEvents p z.reduce := funext reduce
  simp only [Reader.events,typedEqual,first,reduceFunction,blocking]

theorem all_events (p : Parameters) (ops : Instructions α) :
    Reader.events p sourcePlan ops = Reader.events p areaPlan ops :=
  events_congr p ops sourcePlan areaPlan rfl (first_events p ops) (reduce_events p) rfl

theorem reduce_reads (p : Parameters) (tile : Nat) :
    ClosedSource.reduceReads p sourcePlan.reduce tile = ClosedSource.reduceReads p areaPlan.reduce tile := by
  simp only [ClosedSource.reduceReads,reduce_env]
  rfl

theorem source_tape_congr (p : Parameters) (a z : Reader.Program)
    (events : Reader.events p a ClosedSource.symbolic = Reader.events p z ClosedSource.symbolic)
    (first : ClosedSource.firstReads p a = ClosedSource.firstReads p z)
    (reduce : ∀ tile, ClosedSource.reduceReads p a.reduce tile = ClosedSource.reduceReads p z.reduce tile)
    (blocking : a.full = z.full) :
    (do
      if p.h.m=0 ∨ p.h.n=0 then some [] else
        let events ← Reader.events p a ClosedSource.symbolic
        let reads ← ClosedSource.firstReads p a
        some (reads ++ events.first.map (ClosedSource.lowerEvent p) ++
          (List.range (Plan.select p.h).streamTiles).flatMap (ClosedSource.reduceReads p a.reduce) ++
          events.reduce.map (ClosedSource.lowerEvent p) ++ ClosedSource.fullReads p a.full ++
          events.full.map (ClosedSource.lowerEvent p))) =
    (do
      if p.h.m=0 ∨ p.h.n=0 then some [] else
        let events ← Reader.events p z ClosedSource.symbolic
        let reads ← ClosedSource.firstReads p z
        some (reads ++ events.first.map (ClosedSource.lowerEvent p) ++
          (List.range (Plan.select p.h).streamTiles).flatMap (ClosedSource.reduceReads p z.reduce) ++
          events.reduce.map (ClosedSource.lowerEvent p) ++ ClosedSource.fullReads p z.full ++
          events.full.map (ClosedSource.lowerEvent p))) := by
  have reduceFunction : ClosedSource.reduceReads p a.reduce = ClosedSource.reduceReads p z.reduce := funext reduce
  simp only [events,first,reduceFunction,blocking]

theorem source_tape (p : Parameters) :
    ProgramView.sourceTape kernels p = ProgramView.sourceTape areaKernels p := by
  simp only [ProgramView.sourceTape,source_compiles,candidate_compiles,bind,Option.bind]
  exact source_tape_congr p sourcePlan areaPlan (all_events p ClosedSource.symbolic) rfl (reduce_reads p) rfl

end
end VeriTile.Bench.Repairs.StreamK.TileChecked.AreaEquivalence
