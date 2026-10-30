import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.Boundary

namespace VeriTile.Bench.Repairs.BmmChunkBwd.RealTrials
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open Boundary
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

/-- One actual source grid in the existing independent Real interpretation. -/
def one (h : HostCall) (choice : Fin 9) (s : BlockState) : Option BlockState :=
  HostGrid.run (h.source (tuning choice) Bool.false).eraseDType.toAlgKernel
    (HostContract.grid h (tuning choice)) (HostGrid.programs (HostContract.grid h (tuning choice))) s

/-- The same grid on actual flattened memory, with no reinitialization. -/
def flatOne (h : HostCall) (alloc : FlatAlloc) (choice : Fin 9) (s : BlockState) : Option BlockState :=
  HostGrid.run (alloc.flattenKernel (h.source (tuning choice) Bool.false).eraseDType.toAlgKernel)
    (HostContract.grid h (tuning choice)) (HostGrid.programs (HostContract.grid h (tuning choice))) s

def history (h : HostCall) : List (Fin 9) → BlockState → Option BlockState
  | [],s => some s
  | choice::rest,s => (one h choice s).bind (history h rest)

/-- Repeated benchmark launches feed their complete actual physical result
into the next launch. Only the initial direct memory binding is flattened. -/
def flatHistory (h : HostCall) (alloc : FlatAlloc) : List (Fin 9) → BlockState → Option BlockState
  | [],s => some s
  | choice::rest,s => (flatOne h alloc choice s).bind (flatHistory h alloc rest)

def scratch (h : HostCall) (alloc : FlatAlloc) (r : RegionName) (off : Nat) : Prop :=
  r = h.Out ∧ off < alloc.extent h.Out

theorem one_flatten (h : HostCall) (alloc : FlatAlloc) (choice : Fin 9)
    (layout : HostContract.Layout h (tuning choice) alloc) (s : BlockState) :
    flatOne h alloc choice (alloc.flattenState s) = (one h choice s).map alloc.flattenState := by
  unfold flatOne one
  rw [HostContract.source_real]
  exact HostGrid.run_flatten Bool.false _ _ _ _ _ _ alloc layout.disjoint layout.closed layout.bounds s

theorem history_flatten (h : HostCall) (alloc : FlatAlloc) (trials : List (Fin 9))
    (legal : ∀ choice ∈ trials, HostContract.Layout h (tuning choice) alloc) (s : BlockState) :
    flatHistory h alloc trials (alloc.flattenState s) = (history h trials s).map alloc.flattenState := by
  induction trials generalizing s with
  | nil => rfl
  | cons choice rest ih =>
    simp only [flatHistory,history,one_flatten h alloc choice (legal choice (by simp)) s]
    cases he : one h choice s with
    | none => rfl
    | some next =>
      simp only [Option.map_some,Option.bind_some]
      exact ih (fun c hc => legal c (by simp [hc])) next

theorem one_frame (h : HostCall) (alloc : FlatAlloc) (choice : Fin 9)
    (layout : HostContract.Layout h (tuning choice) alloc) (s : BlockState) :
    ∃ t, one h choice s = some t ∧
      ∀ r off, ¬ scratch h alloc r off → t.mem r off = s.mem r off := by
  obtain ⟨t,he,_,hf⟩ := HostGrid.run_correct Bool.false (tuning choice).BM (tuning choice).BN
    (tuning choice).BCS (HostContract.parameters h) (HostContract.tuning_positive choice).2.2
    layout.outputDistinct layout.inputsSeparate (HostContract.grid h (tuning choice))
    (HostGrid.programs (HostContract.grid h (tuning choice))) (HostGrid.programs_nodup _) layout.programWrites s
  refine ⟨t,?_,?_⟩
  · simpa only [one,HostContract.source_real] using he
  · intro r off hx
    apply hf
    intro q hq i hi
    by_cases hr : r = h.Out
    · right
      intro ho
      apply hx
      exact ⟨hr,ho ▸ (layout.bounds q hq s).output i hi⟩
    · exact Or.inl hr

theorem history_frame (h : HostCall) (alloc : FlatAlloc) (trials : List (Fin 9))
    (legal : ∀ choice ∈ trials, HostContract.Layout h (tuning choice) alloc) (s : BlockState) :
    ∃ t, history h trials s = some t ∧
      ∀ r off, ¬ scratch h alloc r off → t.mem r off = s.mem r off := by
  induction trials generalizing s with
  | nil => exact ⟨s,rfl,by intros; rfl⟩
  | cons choice rest ih =>
    obtain ⟨mid,he,hf⟩ := one_frame h alloc choice (legal choice (by simp)) s
    obtain ⟨t,ht,hft⟩ := ih (fun c hc => legal c (by simp [hc])) mid
    refine ⟨t,?_,fun r off hx => (hft r off hx).trans (hf r off hx)⟩
    simpa only [history,he,Option.bind_some] using ht

#axiomsClean history_flatten
#axiomsClean history_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.RealTrials
