import bench.repairs.tritonbench_g.streamk_matmul.Whole001.Reference
import bench.repairs.tritonbench_g.streamk_matmul.Whole001.Full
namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace Reference
open Source Protocol Plan
set_option maxHeartbeats 12000000
noncomputable section
local instance (q r I tile program : Nat) : Decidable (readsScratch q r I tile program) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def positiveI (p : Parameters) (tile : Nat) (ht : tile < (select p.h).streamTiles) :
    0 < (select p.h).iterations :=
  iterations_positive p.h (selected_active p.h (by omega)).2 p.blockK

structure Terminal (p : Parameters) (tile : Nat) (ht : tile < (select p.h).streamTiles) where
  program : Fin p.h.programs
  chunk : Chunk
  occurs : chunk ∈ programChunks (select p.h).streamTiles (select p.h).iterations p.h.programs program.val (positiveI p tile ht)
  tile_eq : chunk.tile (select p.h).iterations=tile
  terminal : chunk.hi%(select p.h).iterations=0

theorem terminal_exists (p : Parameters) (tile : Nat) (ht : tile < (select p.h).streamTiles) :
    Nonempty (Terminal p tile ht) := by
  obtain ⟨pr,c,hpr,hc,he,hn⟩:=initializer_exists (select p.h).streamTiles (select p.h).iterations
    p.h.programs tile (positiveI p tile ht) (selected_active p.h (by omega)).1 ht
  exact ⟨⟨⟨pr,hpr⟩,c,hc,he,hn⟩⟩

def terminal (p : Parameters) (tile : Nat) (ht : tile < (select p.h).streamTiles) : Terminal p tile ht :=
  Classical.choice (terminal_exists p tile ht)

structure Partial (p : Parameters) (tile : Nat) (ht : tile < (select p.h).streamTiles)
    (program : Fin p.h.programs) where
  chunk : Chunk
  occurs : chunk ∈ programChunks (select p.h).streamTiles (select p.h).iterations p.h.programs program.val (positiveI p tile ht)
  tile_eq : chunk.tile (select p.h).iterations=tile
  nonterminal : chunk.hi%(select p.h).iterations≠0

theorem partial_exists (p : Parameters) (tile : Nat) (ht : tile < (select p.h).streamTiles)
    (pr : Fin p.h.programs)
    (hr : readsScratch (select p.h).fullIterations (select p.h).partialIterations (select p.h).iterations tile pr.val) :
    Nonempty (Partial p tile ht pr) := by
  obtain ⟨c,⟨hc,hn,he⟩,hu⟩:=source_reads_initialized_scratch p tile pr.val (by omega) hr
  exact ⟨⟨c,hc,he,hn⟩⟩

def partialChunk (p : Parameters) (tile : Nat) (ht : tile < (select p.h).streamTiles)
    (pr : Fin p.h.programs)
    (hr : readsScratch (select p.h).fullIterations (select p.h).partialIterations (select p.h).iterations tile pr.val) :
    Partial p tile ht pr := Classical.choice (partial_exists p tile ht pr hr)

/-- Closed finite instruction reference. Only initial A/B payloads and integer
partition data enter. It never reads initial or final output/scratch memory. -/
def streamValue (p : Parameters) (ops : Instructions α) (a : A p → α) (b : B p → α)
    (tile : Nat) (ht : tile < (select p.h).streamTiles) (lane : Lane p) : α :=
  let t:=terminal p tile ht
  let initial:=chunkValue p ops a b tile t.chunk.lo t.chunk.hi lane
  (List.finRange p.h.programs).foldl (fun value pr=>
    if hr:readsScratch (select p.h).fullIterations (select p.h).partialIterations (select p.h).iterations tile pr.val then
      let c:=(partialChunk p tile ht pr hr).chunk
      ops.atomicAdd value (chunkValue p ops a b tile c.lo c.hi lane)
    else value) initial

#axiomsClean terminal_exists
#axiomsClean partial_exists
#specNonCircular streamValue avoiding [Source.call,Memory.run,Memory.step,Source.firstWave,Source.reduceWave,Source.fullWave]
#specNonCircular matrixProduct avoiding [Source.call,Memory.run,Memory.step,Source.chunkValue,streamValue]
end
end Reference
namespace Source
open Protocol Plan
set_option maxHeartbeats 12000000
noncomputable section
local instance (q r I tile program : Nat) : Decidable (readsScratch q r I tile program) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

/-- Every unconditional tl.load(partials + slots), including inactive output
lanes, reads an initialized finite slot from the completed first launch. -/
theorem reducer_load_value (p : Parameters) (ops : Instructions α) (initial : State α p exterior)
    (tile : Nat) (ht : tile < (select p.h).streamTiles) (pr : Fin p.h.programs)
    (hr : readsScratch (select p.h).fullIterations (select p.h).partialIterations (select p.h).iterations tile pr.val)
    (lane : Lane p) :
    (Memory.run ops.atomicAdd (firstWave p ops) initial).scratch (scratch p pr lane)=
      let c:=(Reference.partialChunk p tile ht pr hr).chunk
      chunkValue p ops initial.inputA initial.inputB tile c.lo c.hi lane := by
  let c:=Reference.partialChunk p tile ht pr hr
  have h:=first_scratch_value p ops initial (by omega) pr c.chunk c.occurs c.nonterminal lane
  rw [c.tile_eq] at h
  exact h

/-- Full three-launch source result for every stream-prefix lane. This closes
actual current C/scratch memory to the input-only ordered instruction graph. -/
theorem call_stream_value (p : Parameters) (ops : Instructions α) (initial : State α p exterior)
    (tile : Nat) (ht : tile < (select p.h).streamTiles) (lane : Lane p) (ha : active p tile lane) :
    (call p ops initial).output (output p tile lane ha)=
      Reference.streamValue p ops initial.inputA initial.inputB tile ht lane := by
  unfold call
  rw [if_neg (active_nonempty p tile lane ha),full_stream_frame p ops _ tile ht lane ha,
    reduce_output_value p ops.atomicAdd _ tile ht lane ha]
  let t:=Reference.terminal p tile ht
  have hactive:active p (t.chunk.tile (select p.h).iterations) lane:=by rw [t.tile_eq]; exact ha
  have hinit:=first_output_value p ops initial (by omega) t.program t.chunk t.occurs t.terminal lane hactive
  simp only [t.tile_eq] at hinit
  unfold programFold Reference.streamValue
  rw [hinit]
  apply fold_congr
  intro pr hpr z
  by_cases hr:readsScratch (select p.h).fullIterations (select p.h).partialIterations (select p.h).iterations tile pr.val
  · rw [if_pos hr,dif_pos hr,reducer_load_value p ops initial tile ht pr hr lane]
  · rw [if_neg hr,dif_neg hr]

#axiomsClean reducer_load_value
#axiomsClean call_stream_value
end
end Source
end VeriTile.Bench.Repairs.StreamK.Whole001
