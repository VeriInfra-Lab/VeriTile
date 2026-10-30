import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.Operands

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
open Whole001 Source VeriTile.Bench.TritonBenchG.StreamK
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 20000

theorem full_maskA (p : Source.Parameters) (pid : Nat) (lane : Lane p) (k iter : Nat) :
    sourcePlan.full.core.loop.maskA.eval
      (loopEnv sourcePlan.full.core (fullEnv p sourcePlan.full pid) (mlane pid lane k) iter)
      (mlane pid lane k) ↔
    row p (pid+(Plan.select p.h).streamTiles) lane<p.h.m ∧ iter*p.h.bk+k<p.h.k := by
  change _ ∧ ((0+iter)*p.h.bk+k<p.h.k) ↔ _
  simp only [Nat.zero_add]
  rfl

theorem full_maskB (p : Source.Parameters) (pid : Nat) (lane : Lane p) (k iter : Nat) :
    sourcePlan.full.core.loop.maskB.eval
      (loopEnv sourcePlan.full.core (fullEnv p sourcePlan.full pid) (mlane pid lane k) iter)
      (mlane pid lane k) ↔
    iter*p.h.bk+k<p.h.k ∧ col p (pid+(Plan.select p.h).streamTiles) lane<p.h.n := by
  change ((0+iter)*p.h.bk+k<p.h.k) ∧ _ ↔ _
  simp only [Nat.zero_add]
  rfl

theorem full_addressA (p : Source.Parameters) (pid : Nat) (lane : Lane p) (k iter : Nat) :
    sourcePlan.full.core.loop.left.index
      (loopEnv sourcePlan.full.core (fullEnv p sourcePlan.full pid) (mlane pid lane k) iter)
      (loopPointers sourcePlan.full.core (fullEnv p sourcePlan.full pid) (mlane pid lane k) iter)
      (mlane pid lane k) =
    row p (pid+(Plan.select p.h).streamTiles) lane*p.strideAM+(iter*p.h.bk+k)*p.strideAK := by
  change (0+(row p (pid+(Plan.select p.h).streamTiles) lane*p.strideAM+k*p.strideAK))+
    (iter-0)*(0+p.h.bk*p.strideAK)=_
  simp only [Nat.sub_zero]
  ring

theorem full_addressB (p : Source.Parameters) (pid : Nat) (lane : Lane p) (k iter : Nat) :
    sourcePlan.full.core.loop.right.index
      (loopEnv sourcePlan.full.core (fullEnv p sourcePlan.full pid) (mlane pid lane k) iter)
      (loopPointers sourcePlan.full.core (fullEnv p sourcePlan.full pid) (mlane pid lane k) iter)
      (mlane pid lane k) =
    (iter*p.h.bk+k)*p.strideBK+col p (pid+(Plan.select p.h).streamTiles) lane*p.strideBN := by
  change (0+(k*p.strideBK+col p (pid+(Plan.select p.h).streamTiles) lane*p.strideBN))+
    (iter-0)*(0+p.h.bk*p.strideBK)=_
  simp only [Nat.sub_zero]
  ring

theorem full_inputA (p : Source.Parameters) (pid : Nat) (ops : Instructions α)
    (a : A p → α) (lane : Lane p) (k iter : Nat) :
    inputA p sourcePlan.full.core ops a (fullEnv p sourcePlan.full pid) (mlane pid lane k) iter=
      loadA p ops.zero a (row p (pid+(Plan.select p.h).streamTiles) lane) (iter*p.h.bk+k) := by
  simp only [inputA,full_maskA,full_addressA,loadA]
  split
  next active =>
    rw [dif_pos (Geometry.input_mask_safe _ _ _ _ _ _ p.inputA active.1 active.2)]
  next inactive => rfl

theorem full_inputB (p : Source.Parameters) (pid : Nat) (ops : Instructions α)
    (b : B p → α) (lane : Lane p) (k iter : Nat) :
    inputB p sourcePlan.full.core ops b (fullEnv p sourcePlan.full pid) (mlane pid lane k) iter=
      loadB p ops.zero b (iter*p.h.bk+k) (col p (pid+(Plan.select p.h).streamTiles) lane) := by
  simp only [inputB,full_maskB,full_addressB,loadB]
  split
  next active =>
    rw [dif_pos (Geometry.input_mask_safe _ _ _ _ _ _ p.inputB active.1 active.2)]
  next inactive => rfl

theorem full_block (p : Source.Parameters) (pid : Nat) (ops : Instructions α)
    (a : A p → α) (b : B p → α) (lane : Lane p) (iter : Nat) (seed : α)
    (range : iter<(Plan.select p.h).iterations) :
    dotBlock p sourcePlan.full.core ops a b (fullEnv p sourcePlan.full pid) lane iter seed=
      block p ops a b (pid+(Plan.select p.h).streamTiles) iter lane seed := by
  unfold dotBlock block
  rw [Nat.mod_eq_of_lt range]
  apply Checked.Index.fold_same
  intro group hg acc
  congr 1
  · funext j
    simpa only [Nat.add_assoc] using full_inputA p pid ops a lane (group*8+j.val) iter
  · funext j
    simpa only [Nat.add_assoc] using full_inputB p pid ops b lane (group*8+j.val) iter

theorem full_value (p : Source.Parameters) (pid : Nat) (ops : Instructions α)
    (a : A p → α) (b : B p → α) (lane : Lane p) :
    dotValue p sourcePlan.full.core ops a b (fullEnv p sourcePlan.full pid) lane=
      chunkValue p ops a b (pid+(Plan.select p.h).streamTiles) 0 (Plan.select p.h).iterations lane := by
  change (List.range' 0 (ceilDiv p.h.k p.h.bk-0)).foldl _ ops.zero=
    (List.range' 0 ((Plan.select p.h).iterations-0)).foldl _ ops.zero
  rw [←Plan.selected_iterations]
  apply Checked.Index.fold_same
  intro iter hi acc
  obtain ⟨offset,hoff,heq⟩ := List.mem_range'.mp hi
  simp only [Nat.one_mul] at heq
  exact full_block p pid ops a b lane iter acc (by omega)

#axiomsClean full_addressA
#axiomsClean full_addressB
#axiomsClean full_inputA
#axiomsClean full_inputB
#axiomsClean full_block
#axiomsClean full_value
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
