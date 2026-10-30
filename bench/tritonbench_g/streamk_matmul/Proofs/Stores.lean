import bench.tritonbench_g.streamk_matmul.Proofs.FullOperands

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
open Whole001 Source
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 20000

theorem mac_terminal (p : Source.Parameters) (program : Nat) (c : Protocol.Chunk) :
    sourcePlan.mac.terminal.eval (macEnv p sourcePlan.mac program c) ⟨program,0,0,0⟩ ↔
    c.hi%(Plan.select p.h).iterations=0 := by rfl

theorem mac_output_mask (p : Source.Parameters) (program : Nat) (c : Protocol.Chunk) (lane : Lane p) :
    sourcePlan.mac.mask.eval (tileEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane 0))
      (mlane program lane 0) ↔ active p (c.tile (Plan.select p.h).iterations) lane := by rfl

theorem mac_output_index (p : Source.Parameters) (program : Nat) (c : Protocol.Chunk) (lane : Lane p)
    (ha : active p (c.tile (Plan.select p.h).iterations) lane) :
    sourcePlan.mac.output.index (tileEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane 0))
      zeroPointers (mlane program lane 0) = (output p (c.tile (Plan.select p.h).iterations) lane ha).val := by
  change 0+(row p (c.tile (Plan.select p.h).iterations) lane*max p.h.n 1+col p (c.tile (Plan.select p.h).iterations) lane*1)=row p (c.tile (Plan.select p.h).iterations) lane*p.h.n+col p (c.tile (Plan.select p.h).iterations) lane
  have hn : max p.h.n 1=p.h.n := max_eq_left (by have := ha.2; omega)
  rw [hn]
  omega

theorem mac_scratch_index (p : Source.Parameters) (program : Fin p.h.programs) (c : Protocol.Chunk) (lane : Lane p) :
    let e := tileEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program.val c) (mlane program.val lane 0)
    sourcePlan.mac.scratch.index (put e "slots" (sourcePlan.mac.slots.eval e (mlane program.val lane 0)))
      zeroPointers (mlane program.val lane 0) = (scratch p program lane).val := by
  change 0+(program.val*p.h.bm*p.h.bn+lane.1.val*p.h.bn+lane.2.val)=_
  simp only [Nat.zero_add,scratch]

theorem mac_events (p : Source.Parameters) (program : Fin p.h.programs) (c : Protocol.Chunk)
    (ops : Instructions α) (hI : 0<(Plan.select p.h).iterations)
    (bound : c.hi≤Protocol.boundary (Plan.select p.h).iterations c.lo) :
    macEvents p sourcePlan.mac ops program c=macLoop p ops program c := by
  have value : ∀ lane, (fun a b => dotValue p sourcePlan.mac.core ops a b (macEnv p sourcePlan.mac program.val c) lane)=
      fun a b => chunkValue p ops a b (c.tile (Plan.select p.h).iterations) c.lo c.hi lane := by
    intro lane
    funext a b
    exact mac_value p program.val c ops a b lane hI bound
  unfold macEvents macLoop
  dsimp only
  simp only [mac_terminal,value]
  by_cases terminal : c.hi%(Plan.select p.h).iterations=0
  · simp only [if_pos terminal,storeTile]
    apply List.flatMap_congr
    intro lane hl
    simp only [mac_output_mask]
    split
    next ha =>
      rw [mac_output_index p program.val c lane ha]
      simp only [outputEvent,dif_pos (output p _ lane ha).isLt]
    next => rfl
  · simp only [if_neg terminal]
    rw [←List.flatMap_pure_eq_map]
    apply List.flatMap_congr
    intro lane hl
    rw [mac_scratch_index]
    simp only [scratchEvent,dif_pos (scratch p program lane).isLt]
    rfl

theorem full_output_mask (p : Source.Parameters) (pid : Nat) (lane : Lane p) :
    let l := mlane pid lane 0
    let e := tileEnv sourcePlan.full.core (fullEnv p sourcePlan.full pid) l
    let e := put e "rm" (sourcePlan.full.rematM.eval e l)
    let e := put e "rn" (sourcePlan.full.rematN.eval e l)
    sourcePlan.full.mask.eval e l ↔ active p (pid+(Plan.select p.h).streamTiles) lane := by rfl

theorem full_output_index (p : Source.Parameters) (pid : Nat) (lane : Lane p)
    (ha : active p (pid+(Plan.select p.h).streamTiles) lane) :
    let l := mlane pid lane 0
    let e := tileEnv sourcePlan.full.core (fullEnv p sourcePlan.full pid) l
    let e := put e "rm" (sourcePlan.full.rematM.eval e l)
    let e := put e "rn" (sourcePlan.full.rematN.eval e l)
    sourcePlan.full.output.index e zeroPointers l = (output p (pid+(Plan.select p.h).streamTiles) lane ha).val := by
  change 0+(row p (pid+(Plan.select p.h).streamTiles) lane*max p.h.n 1+col p (pid+(Plan.select p.h).streamTiles) lane*1)=row p (pid+(Plan.select p.h).streamTiles) lane*p.h.n+col p (pid+(Plan.select p.h).streamTiles) lane
  have hn : max p.h.n 1=p.h.n := max_eq_left (by have := ha.2; omega)
  rw [hn]
  omega

theorem full_events (p : Source.Parameters) (pid : Nat) (ops : Instructions α) :
    fullEvents p sourcePlan.full ops pid=
      storeTile p (fun lane a b => ops.halfRoundTrip
        (chunkValue p ops a b (pid+(Plan.select p.h).streamTiles) 0 (Plan.select p.h).iterations lane))
        (pid+(Plan.select p.h).streamTiles) := by
  unfold fullEvents storeTile
  apply List.flatMap_congr
  intro lane hl
  dsimp only
  simp only [full_output_mask]
  split
  next ha =>
    rw [full_output_index p pid lane ha]
    simp only [outputEvent,dif_pos (output p _ lane ha).isLt,full_value]
  next => rfl

#axiomsClean mac_events
#axiomsClean full_events
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
