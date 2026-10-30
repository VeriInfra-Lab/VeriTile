import bench.tritonbench_g.streamk_matmul.Proofs.SourceBridge

/-! Direct bounds for the quotation's actual pointer/mask expressions. These
include source loads in lanes without output events. They are ordinary helper
lemmas for the pending shared source-access judgment, not assumed safety. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
open Whole001 Source
noncomputable section
set_option maxHeartbeats 4000000

theorem mac_loadA_bound (p : Parameters) (program : Nat) (c : Protocol.Chunk)
    (lane : Lane p) (k iter : Nat) (positive : 0<(Plan.select p.h).iterations)
    (chunk : c.hi≤Protocol.boundary (Plan.select p.h).iterations c.lo)
    (inside : c.lo ≤ iter ∧ iter < c.hi)
    (active : sourcePlan.mac.core.loop.maskA.eval
      (loopEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane k) iter)
      (mlane program lane k)) :
    sourcePlan.mac.core.loop.left.index
      (loopEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane k) iter)
      (loopPointers sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane k) iter)
      (mlane program lane k)<p.h.m*p.h.k := by
  rw [mac_addressA p program c lane k iter positive chunk inside]
  have on := (mac_maskA p program c lane k iter).mp active
  exact Geometry.input_mask_safe _ _ _ _ _ _ p.inputA on.1 on.2

theorem mac_loadB_bound (p : Parameters) (program : Nat) (c : Protocol.Chunk)
    (lane : Lane p) (k iter : Nat) (positive : 0<(Plan.select p.h).iterations)
    (chunk : c.hi≤Protocol.boundary (Plan.select p.h).iterations c.lo)
    (inside : c.lo ≤ iter ∧ iter < c.hi)
    (active : sourcePlan.mac.core.loop.maskB.eval
      (loopEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane k) iter)
      (mlane program lane k)) :
    sourcePlan.mac.core.loop.right.index
      (loopEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane k) iter)
      (loopPointers sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane k) iter)
      (mlane program lane k)<p.h.k*p.h.n := by
  rw [mac_addressB p program c lane k iter positive chunk inside]
  have on := (mac_maskB p program c lane k iter).mp active
  exact Geometry.input_mask_safe _ _ _ _ _ _ p.inputB on.1 on.2

theorem full_loadA_bound (p : Parameters) (pid : Nat) (lane : Lane p) (k iter : Nat)
    (active : sourcePlan.full.core.loop.maskA.eval
      (loopEnv sourcePlan.full.core (fullEnv p sourcePlan.full pid) (mlane pid lane k) iter)
      (mlane pid lane k)) :
    sourcePlan.full.core.loop.left.index
      (loopEnv sourcePlan.full.core (fullEnv p sourcePlan.full pid) (mlane pid lane k) iter)
      (loopPointers sourcePlan.full.core (fullEnv p sourcePlan.full pid) (mlane pid lane k) iter)
      (mlane pid lane k)<p.h.m*p.h.k := by
  rw [full_addressA]
  have on := (full_maskA p pid lane k iter).mp active
  exact Geometry.input_mask_safe _ _ _ _ _ _ p.inputA on.1 on.2

theorem full_loadB_bound (p : Parameters) (pid : Nat) (lane : Lane p) (k iter : Nat)
    (active : sourcePlan.full.core.loop.maskB.eval
      (loopEnv sourcePlan.full.core (fullEnv p sourcePlan.full pid) (mlane pid lane k) iter)
      (mlane pid lane k)) :
    sourcePlan.full.core.loop.right.index
      (loopEnv sourcePlan.full.core (fullEnv p sourcePlan.full pid) (mlane pid lane k) iter)
      (loopPointers sourcePlan.full.core (fullEnv p sourcePlan.full pid) (mlane pid lane k) iter)
      (mlane pid lane k)<p.h.k*p.h.n := by
  rw [full_addressB]
  have on := (full_maskB p pid lane k iter).mp active
  exact Geometry.input_mask_safe _ _ _ _ _ _ p.inputB on.1 on.2

theorem mac_output_bound (p : Parameters) (program : Nat) (c : Protocol.Chunk) (lane : Lane p)
    (active : sourcePlan.mac.mask.eval
      (tileEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane 0))
      (mlane program lane 0)) :
    sourcePlan.mac.output.index
      (tileEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane 0))
      zeroPointers (mlane program lane 0)<p.h.m*p.h.n := by
  rw [mac_output_index p program c lane ((mac_output_mask p program c lane).mp active)]
  exact (output p _ lane _).isLt

theorem mac_scratch_bound (p : Parameters) (program : Fin p.h.programs)
    (c : Protocol.Chunk) (lane : Lane p) :
    let e := tileEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program.val c) (mlane program.val lane 0)
    sourcePlan.mac.scratch.index (put e "slots" (sourcePlan.mac.slots.eval e (mlane program.val lane 0)))
      zeroPointers (mlane program.val lane 0)<p.h.programs*p.h.bm*p.h.bn := by
  dsimp only
  rw [mac_scratch_index]
  exact (scratch p program lane).isLt

/-- No output mask premise: every source reducer lane performs this load. -/
theorem reduce_load_bound (p : Parameters) (tile : Nat) (program : Fin p.h.programs) (lane : Lane p) :
    sourcePlan.reduce.load.index (reduceEnv p sourcePlan.reduce tile program.val lane)
      zeroPointers (mlane tile lane 0)<p.h.programs*p.h.bm*p.h.bn := by
  rw [reduce_scratch_index]
  exact (scratch p program lane).isLt

theorem reduce_output_bound (p : Parameters) (tile program : Nat) (lane : Lane p)
    (active : sourcePlan.reduce.atomicMask.eval
      (reduceEnv p sourcePlan.reduce tile program lane) (mlane tile lane 0)) :
    sourcePlan.reduce.output.index (reduceEnv p sourcePlan.reduce tile program lane)
      zeroPointers (mlane tile lane 0)<p.h.m*p.h.n := by
  rw [reduce_output_index p tile program lane ((reduce_mask p tile program lane).mp active)]
  exact (output p tile lane _).isLt

theorem full_output_bound (p : Parameters) (pid : Nat) (lane : Lane p)
    (active : Source.active p (pid+(Plan.select p.h).streamTiles) lane) :
    let l := mlane pid lane 0
    let e := tileEnv sourcePlan.full.core (fullEnv p sourcePlan.full pid) l
    let e := put e "rm" (sourcePlan.full.rematM.eval e l)
    let e := put e "rn" (sourcePlan.full.rematN.eval e l)
    sourcePlan.full.output.index e zeroPointers l<p.h.m*p.h.n := by
  dsimp only
  rw [full_output_index p pid lane active]
  exact (output p _ lane active).isLt

#axiomsClean mac_loadA_bound
#axiomsClean mac_loadB_bound
#axiomsClean full_loadA_bound
#axiomsClean full_loadB_bound
#axiomsClean mac_output_bound
#axiomsClean mac_scratch_bound
#axiomsClean reduce_load_bound
#axiomsClean reduce_output_bound
#axiomsClean full_output_bound
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
