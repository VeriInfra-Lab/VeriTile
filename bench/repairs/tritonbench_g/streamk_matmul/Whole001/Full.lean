import bench.repairs.tritonbench_g.streamk_matmul.Whole001.Reduction
namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace Source
open Protocol Plan
set_option maxHeartbeats 12000000
noncomputable section

theorem full_output_member (p : Parameters) (ops : Instructions α)
    (tile : Nat) (ht : tile < tiles p.h) (hs : (select p.h).streamTiles ≤ tile)
    (lane : Lane p) (ha : active p tile lane) :
    Memory.Event.storeOutput (output p tile lane ha)
      (fun a b=>ops.halfRoundTrip (chunkValue p ops a b tile 0 (select p.h).iterations lane)) ∈ fullWave p ops := by
  unfold fullWave
  apply List.mem_flatMap.mpr
  have hsum:=selected_partition p.h
  refine ⟨tile-(select p.h).streamTiles,List.mem_range.mpr (by omega),?_⟩
  have he:tile-(select p.h).streamTiles+(select p.h).streamTiles=tile:=by omega
  rw [he]
  unfold storeTile
  apply List.mem_flatMap.mpr
  refine ⟨lane,lane_mem p lane,?_⟩
  rw [dif_pos ha]
  exact List.mem_singleton_self _

theorem full_output_unique (p : Parameters) (ops : Instructions α)
    (tile : Nat) (ht : tile < tiles p.h) (lane : Lane p) (ha : active p tile lane)
    (e : Event α p) (he : e ∈ fullWave p ops) (hw : Memory.WritesOutput (output p tile lane ha) e) :
    e=Memory.Event.storeOutput (output p tile lane ha)
      (fun a b=>ops.halfRoundTrip (chunkValue p ops a b tile 0 (select p.h).iterations lane)) := by
  unfold fullWave at he
  obtain ⟨pid,hpid,he⟩:=List.mem_flatMap.mp he
  unfold storeTile at he
  obtain ⟨other,hother,he⟩:=List.mem_flatMap.mp he
  split at he
  next hactive =>
    simp only [List.mem_singleton] at he
    subst e
    change output p (pid+(select p.h).streamTiles) other hactive=output p tile lane ha at hw
    have hpid':pid < (select p.h).blockingTiles:=List.mem_range.mp hpid
    have hsum:=selected_partition p.h
    have bound:pid+(select p.h).streamTiles < tiles p.h:=by omega
    obtain ⟨hteq,heq⟩:=output_address_unique p _ tile bound ht other lane hactive ha hw
    subst other
    simp only [hteq]
  next hactive => simp at he

theorem full_output_value (p : Parameters) (ops : Instructions α) (initial : State α p exterior)
    (tile : Nat) (ht : tile < tiles p.h) (hs : (select p.h).streamTiles ≤ tile)
    (lane : Lane p) (ha : active p tile lane) :
    (Memory.run ops.atomicAdd (fullWave p ops) initial).output (output p tile lane ha)=
      ops.halfRoundTrip (chunkValue p ops initial.inputA initial.inputB tile 0 (select p.h).iterations lane) := by
  apply Memory.output_writer_result ops.atomicAdd (fullWave p ops) initial (output p tile lane ha)
    (fun a b=>ops.halfRoundTrip (chunkValue p ops a b tile 0 (select p.h).iterations lane))
  · exact full_output_member p ops tile ht hs lane ha
  · exact full_output_unique p ops tile ht lane ha

/-- Full suffix writes cannot change any stream-prefix cell. -/
theorem full_stream_frame (p : Parameters) (ops : Instructions α) (initial : State α p exterior)
    (tile : Nat) (ht : tile < (select p.h).streamTiles) (lane : Lane p) (ha : active p tile lane) :
    (Memory.run ops.atomicAdd (fullWave p ops) initial).output (output p tile lane ha)=
      initial.output (output p tile lane ha) := by
  apply Memory.run_output_frame
  intro e he hw
  unfold fullWave at he
  obtain ⟨pid,hpid,he⟩:=List.mem_flatMap.mp he
  unfold storeTile at he
  obtain ⟨other,hother,he⟩:=List.mem_flatMap.mp he
  split at he
  next hactive =>
    simp only [List.mem_singleton] at he
    subst e
    change output p (pid+(select p.h).streamTiles) other hactive=output p tile lane ha at hw
    have hpid':pid < (select p.h).blockingTiles:=List.mem_range.mp hpid
    have hsum:=selected_partition p.h
    have bound:pid+(select p.h).streamTiles < tiles p.h:=by omega
    have htarget:=lt_of_lt_of_le ht (selected_bounded p.h)
    have eq:=(output_address_unique p _ tile bound htarget other lane hactive ha hw).1
    omega
  next hactive => simp at he

theorem active_nonempty (p : Parameters) (tile : Nat) (lane : Lane p) (ha : active p tile lane) :
    ¬(p.h.m=0 ∨ p.h.n=0) := by
  have hm:=ha.1;have hn:=ha.2
  omega

/-- Whole host result for every classical suffix tile, including K=0. The
original explicit FP16 roundtrip is part of the returned FP32 value. -/
theorem call_full_value (p : Parameters) (ops : Instructions α) (initial : State α p exterior)
    (tile : Nat) (ht : tile < tiles p.h) (hs : (select p.h).streamTiles ≤ tile)
    (lane : Lane p) (ha : active p tile lane) :
    (call p ops initial).output (output p tile lane ha)=
      ops.halfRoundTrip (chunkValue p ops initial.inputA initial.inputB tile 0 (select p.h).iterations lane) := by
  unfold call
  rw [if_neg (active_nonempty p tile lane ha),full_output_value p ops _ tile ht hs lane ha]
  have hr:=Memory.run_inputs ops.atomicAdd (reduceWave p)
    (Memory.run ops.atomicAdd (firstWave p ops) initial)
  have hf:=Memory.run_inputs ops.atomicAdd (firstWave p ops) initial
  rw [hr.1,hr.2.1,hf.1,hf.2.1]

#axiomsClean full_output_value
#axiomsClean full_stream_frame
#axiomsClean call_full_value
end
end Source
end VeriTile.Bench.Repairs.StreamK.Whole001
