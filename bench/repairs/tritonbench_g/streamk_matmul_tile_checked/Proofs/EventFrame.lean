import VeriTile.Triton.Concurrent.EventFrame
import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.EventWindows

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.EventIO
open Whole001 Source VeriTile.Triton Concurrent
noncomputable section
set_option maxHeartbeats 5000000

theorem lower_writes (p : Parameters) (observe : V → O) (e : Source.Event Expr p)
    (scratchActive : ∀ i value,e=.storeScratch i value → 0<(Plan.select p.h).streamTiles)
    (region : Nat) (address : Int) (write : (ClosedSource.lowerEvent p e).WritesMem region address) :
    (io p observe).ScalarWritten region address := by
  cases e with
  | storeOutput i value =>
    obtain ⟨rfl,rfl⟩ := write
    exact c_written p observe i
  | storeScratch i value =>
    obtain ⟨rfl,rfl⟩ := write
    exact s_written p observe i (scratchActive i value rfl)
  | atomic i j =>
    obtain ⟨rfl,rfl⟩ := write
    exact c_written p observe i

theorem first_writes (p : Parameters) (observe : V → O) (event : Concurrent.Event)
    (member : event∈(firstWave p ClosedSource.symbolic).map (ClosedSource.lowerEvent p))
    (region : Nat) (address : Int) (write : event.WritesMem region address) :
    (io p observe).ScalarWritten region address := by
  obtain ⟨e,he,rfl⟩ := List.mem_map.mp member
  have active : 0<(Plan.select p.h).streamTiles := by
    by_contra inactive
    simp [firstWave,inactive] at he
  exact lower_writes p observe e (fun _ _ _=>active) region address write

theorem reduce_writes (p : Parameters) (observe : V → O) (event : Concurrent.Event)
    (member : event∈(reduceWave p).map (ClosedSource.lowerEvent p))
    (region : Nat) (address : Int) (write : event.WritesMem region address) :
    (io p observe).ScalarWritten region address := by
  obtain ⟨e,he,rfl⟩ := List.mem_map.mp member
  obtain ⟨pair,_,rfl⟩ := List.mem_map.mp he
  obtain ⟨rfl,rfl⟩ := write
  exact c_written p observe pair.1

theorem full_writes (p : Parameters) (observe : V → O) (event : Concurrent.Event)
    (member : event∈(fullWave p ClosedSource.symbolic).map (ClosedSource.lowerEvent p))
    (region : Nat) (address : Int) (write : event.WritesMem region address) :
    (io p observe).ScalarWritten region address := by
  obtain ⟨e,he,rfl⟩ := List.mem_map.mp member
  obtain ⟨pid,_,he⟩ := List.mem_flatMap.mp he
  obtain ⟨lane,_,he⟩ := List.mem_flatMap.mp he
  split at he
  · simp only [List.mem_cons,List.not_mem_nil,or_false] at he
    subst e
    obtain ⟨rfl,rfl⟩ := write
    exact c_written p observe _
  · simp at he

theorem readOnly_no_mem (event : Concurrent.Event) (h : ClosedSource.ReadOnly event)
    (region : Nat) (address : Int) : ¬event.WritesMem region address := by
  cases event <;> simp_all [ClosedSource.ReadOnly,Event.WritesMem]

theorem readOnly_no_word (event : Concurrent.Event) (h : ClosedSource.ReadOnly event)
    (region : Nat) (address : Int) : ¬event.WritesWord region address := by
  cases event <;> simp_all [ClosedSource.ReadOnly,Event.WritesWord]

theorem lower_no_word (p : Parameters) (event : Source.Event Expr p)
    (region : Nat) (address : Int) : ¬(ClosedSource.lowerEvent p event).WritesWord region address := by
  cases event <;> simp [ClosedSource.lowerEvent,Event.WritesWord]

theorem complete_writes (p : Parameters) (observe : V → O) (event : Concurrent.Event)
    (member : event∈ClosedSource.completeTape p) (region : Nat) (address : Int)
    (write : event.WritesMem region address) : (io p observe).ScalarWritten region address := by
  unfold ClosedSource.completeTape at member
  split at member
  · simp at member
  · simp only [List.mem_append] at member
    rcases member with ((((h|h)|h)|h)|h)|h
    · exact False.elim (readOnly_no_mem event (ClosedSource.firstReadTape_readOnly p event h) region address write)
    · exact first_writes p observe event h region address write
    · exact False.elim (readOnly_no_mem event
        (ClosedSource.flatMap_readOnly _ _ (ClosedSource.reduceReads_readOnly p Reader.sourcePlan.reduce) event h)
        region address write)
    · exact reduce_writes p observe event h region address write
    · exact False.elim (readOnly_no_mem event (ClosedSource.fullReads_readOnly p Reader.sourcePlan.full event h)
        region address write)
    · exact full_writes p observe event h region address write

theorem complete_no_word (p : Parameters) (event : Concurrent.Event)
    (member : event∈ClosedSource.completeTape p) (region : Nat) (address : Int) :
    ¬event.WritesWord region address := by
  unfold ClosedSource.completeTape at member
  split at member
  · simp at member
  · simp only [List.mem_append] at member
    rcases member with ((((h|h)|h)|h)|h)|h
    · exact readOnly_no_word event (ClosedSource.firstReadTape_readOnly p event h) region address
    · obtain ⟨e,_,rfl⟩ := List.mem_map.mp h
      exact lower_no_word p e region address
    · exact readOnly_no_word event
        (ClosedSource.flatMap_readOnly _ _ (ClosedSource.reduceReads_readOnly p Reader.sourcePlan.reduce) event h)
        region address
    · obtain ⟨e,_,rfl⟩ := List.mem_map.mp h
      exact lower_no_word p e region address
    · exact readOnly_no_word event (ClosedSource.fullReads_readOnly p Reader.sourcePlan.full event h) region address
    · obtain ⟨e,_,rfl⟩ := List.mem_map.mp h
      exact lower_no_word p e region address

theorem complete_frame (p : Parameters) (observe : V → O) (backend : Concurrent.Backend V)
    (s : Concurrent.State V) : (io p observe).Frame s (Concurrent.run backend (ClosedSource.completeTape p) s) :=
  EventProgramIO.frame_of_writes (io p observe) backend (.tape (ClosedSource.completeTape p)) s _ _
    (tape_runs backend _ s) (complete_writes p observe)
    (fun e he r a hw=>False.elim (complete_no_word p e he r a hw))

#axiomsClean complete_writes
#axiomsClean complete_no_word
#axiomsClean complete_frame
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.EventIO
