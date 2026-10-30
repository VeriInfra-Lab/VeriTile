import bench.tritonbench_g.streamk_matmul.Proofs.Control
import bench.tritonbench_g.streamk_matmul.Proofs.Index

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
open Whole001 Source
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 20000

/-- The interpreter's affine pointer expression is exactly repeated execution
of the source `A += step_a` / `B += step_b` instruction, including its final
one-past pointer. It is not a pointer value supplied by a caller. -/
theorem affine_pointer_loop (base step count : Nat) :
    (List.range count).foldl (fun pointer _ => pointer+step) base=base+count*step := by
  induction count with
  | zero => simp
  | succ count ih =>
    rw [List.range_succ,List.foldl_append,ih]
    simp only [List.foldl_cons,List.foldl_nil,Nat.succ_mul,Nat.add_assoc]

theorem remainder_progress (I lo hi iter : Nat) (hI : 0<I)
    (bound : hi≤Protocol.boundary I lo) (range : lo≤ iter ∧ iter<hi) :
    lo%I+(iter-lo)=iter%I := by
  have same := Protocol.same_tile I lo hi iter hI bound range.1 range.2
  have a := Nat.div_add_mod lo I
  have b := Nat.div_add_mod iter I
  rw [same] at b
  omega

theorem mac_addressA (p : Source.Parameters) (program : Nat) (c : Protocol.Chunk)
    (lane : Lane p) (k iter : Nat) (hI : 0<(Plan.select p.h).iterations)
    (bound : c.hi≤Protocol.boundary (Plan.select p.h).iterations c.lo)
    (range : c.lo≤ iter ∧ iter<c.hi) :
    sourcePlan.mac.core.loop.left.index
      (loopEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane k) iter)
      (loopPointers sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane k) iter)
      (mlane program lane k) =
    Source.row p (c.tile (Plan.select p.h).iterations) lane*p.strideAM+
      ((iter%(Plan.select p.h).iterations)*p.h.bk+k)*p.strideAK := by
  change (0+(Source.row p (c.tile (Plan.select p.h).iterations) lane*p.strideAM+k*p.strideAK))+
      p.h.bk*p.strideAK*(c.lo%(Plan.select p.h).iterations)+
      (iter-c.lo)*(0+p.h.bk*p.strideAK)=_
  have progress := remainder_progress _ _ _ _ hI bound range
  calc
    _ = Source.row p (c.tile (Plan.select p.h).iterations) lane*p.strideAM+
        ((c.lo%(Plan.select p.h).iterations+(iter-c.lo))*p.h.bk+k)*p.strideAK := by ring
    _ = _ := by rw [progress]

theorem mac_addressB (p : Source.Parameters) (program : Nat) (c : Protocol.Chunk)
    (lane : Lane p) (k iter : Nat) (hI : 0<(Plan.select p.h).iterations)
    (bound : c.hi≤Protocol.boundary (Plan.select p.h).iterations c.lo)
    (range : c.lo≤ iter ∧ iter<c.hi) :
    sourcePlan.mac.core.loop.right.index
      (loopEnv sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane k) iter)
      (loopPointers sourcePlan.mac.core (macEnv p sourcePlan.mac program c) (mlane program lane k) iter)
      (mlane program lane k) =
      ((iter%(Plan.select p.h).iterations)*p.h.bk+k)*p.strideBK+
      Source.col p (c.tile (Plan.select p.h).iterations) lane*p.strideBN := by
  change (0+(k*p.strideBK+Source.col p (c.tile (Plan.select p.h).iterations) lane*p.strideBN))+
      p.h.bk*p.strideBK*(c.lo%(Plan.select p.h).iterations)+
      (iter-c.lo)*(0+p.h.bk*p.strideBK)=_
  have progress := remainder_progress _ _ _ _ hI bound range
  calc
    _ = ((c.lo%(Plan.select p.h).iterations+(iter-c.lo))*p.h.bk+k)*p.strideBK+
        Source.col p (c.tile (Plan.select p.h).iterations) lane*p.strideBN := by ring
    _ = _ := by rw [progress]

theorem mac_inputA (p : Source.Parameters) (program : Nat) (c : Protocol.Chunk)
    (ops : Instructions α) (a : A p → α) (lane : Lane p) (k iter : Nat)
    (hI : 0<(Plan.select p.h).iterations)
    (bound : c.hi≤Protocol.boundary (Plan.select p.h).iterations c.lo)
    (range : c.lo≤ iter ∧ iter<c.hi) :
    inputA p sourcePlan.mac.core ops a (macEnv p sourcePlan.mac program c) (mlane program lane k) iter=
      loadA p ops.zero a (row p (c.tile (Plan.select p.h).iterations) lane)
        ((iter%(Plan.select p.h).iterations)*p.h.bk+k) := by
  simp only [inputA,mac_maskA,mac_addressA p program c lane k iter hI bound range,loadA]
  split
  next active =>
    rw [dif_pos (Geometry.input_mask_safe _ _ _ _ _ _ p.inputA active.1 active.2)]
  next inactive => rfl

theorem mac_inputB (p : Source.Parameters) (program : Nat) (c : Protocol.Chunk)
    (ops : Instructions α) (b : B p → α) (lane : Lane p) (k iter : Nat)
    (hI : 0<(Plan.select p.h).iterations)
    (bound : c.hi≤Protocol.boundary (Plan.select p.h).iterations c.lo)
    (range : c.lo≤ iter ∧ iter<c.hi) :
    inputB p sourcePlan.mac.core ops b (macEnv p sourcePlan.mac program c) (mlane program lane k) iter=
      loadB p ops.zero b ((iter%(Plan.select p.h).iterations)*p.h.bk+k)
        (col p (c.tile (Plan.select p.h).iterations) lane) := by
  simp only [inputB,mac_maskB,mac_addressB p program c lane k iter hI bound range,loadB]
  split
  next active =>
    rw [dif_pos (Geometry.input_mask_safe _ _ _ _ _ _ p.inputB active.1 active.2)]
  next inactive => rfl

theorem mac_block (p : Source.Parameters) (program : Nat) (c : Protocol.Chunk)
    (ops : Instructions α) (a : A p → α) (b : B p → α) (lane : Lane p) (iter : Nat) (seed : α)
    (hI : 0<(Plan.select p.h).iterations)
    (bound : c.hi≤Protocol.boundary (Plan.select p.h).iterations c.lo)
    (range : c.lo≤ iter ∧ iter<c.hi) :
    dotBlock p sourcePlan.mac.core ops a b (macEnv p sourcePlan.mac program c) lane iter seed=
      block p ops a b (c.tile (Plan.select p.h).iterations) iter lane seed := by
  unfold dotBlock block
  apply Checked.Index.fold_same
  intro group hg acc
  congr 1
  · funext j
    simpa only [Nat.add_assoc] using mac_inputA p program c ops a lane (group*8+j.val) iter hI bound range
  · funext j
    simpa only [Nat.add_assoc] using mac_inputB p program c ops b lane (group*8+j.val) iter hI bound range

theorem mac_value (p : Source.Parameters) (program : Nat) (c : Protocol.Chunk)
    (ops : Instructions α) (a : A p → α) (b : B p → α) (lane : Lane p)
    (hI : 0<(Plan.select p.h).iterations)
    (bound : c.hi≤Protocol.boundary (Plan.select p.h).iterations c.lo) :
    dotValue p sourcePlan.mac.core ops a b (macEnv p sourcePlan.mac program c) lane=
      chunkValue p ops a b (c.tile (Plan.select p.h).iterations) c.lo c.hi lane := by
  change (List.range' c.lo (c.hi-c.lo)).foldl _ ops.zero=(List.range' c.lo (c.hi-c.lo)).foldl _ ops.zero
  apply Checked.Index.fold_same
  intro iter hi acc
  obtain ⟨offset,hoff,heq⟩ := List.mem_range'.mp hi
  simp only [Nat.one_mul] at heq
  exact mac_block p program c ops a b lane iter acc hI bound ⟨by omega,by omega⟩

#axiomsClean affine_pointer_loop
#axiomsClean remainder_progress
#axiomsClean mac_addressA
#axiomsClean mac_addressB
#axiomsClean mac_inputA
#axiomsClean mac_inputB
#axiomsClean mac_block
#axiomsClean mac_value
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
