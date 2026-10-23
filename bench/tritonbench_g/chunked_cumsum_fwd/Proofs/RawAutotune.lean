import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawPhysical

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBoundary
open VeriTile.Triton FP.Structural Math Program RawObserved Backend RawBackend
noncomputable section
set_option maxHeartbeats 3000000

structure HeadBlock where
  size : Nat
  legal : size ∈ [1,2,4,8,16,32,64]

def reblock (c : Whole.Host) (b : HeadBlock) : Whole.Host :=
  { c with BH := b.size, legalHeadBlock := b.legal }

/-- Only autotuner choices and a complete schedule; no kernel result. Models
and ordered trees may differ by profile, including the audited bias order. -/
structure LaunchPlan (α : Type) (c : Whole.Host) where
  block : HeadBlock
  model : NativeModel α
  scan : PrefixPlan block.size c.BC
  order : List Launch
  complete : order.Perm (reblock c block).grid

def runLaunch (v : Views) (c : Whole.Host) (hasBias useSoftplus : Bool) (lower upper : α)
    (p : LaunchPlan α c) (memory : PhysicalMemory α) : Option (PhysicalMemory α) :=
  runPhysical p.model (reblock c p.block) p.scan v hasBias useSoftplus lower upper p.order memory

def runTrials (v : Views) (c : Whole.Host) (hasBias useSoftplus : Bool) (lower upper : α) :
    List (LaunchPlan α c) → PhysicalMemory α → Option (PhysicalMemory α)
  | [],memory => some memory
  | p::rest,memory => (runLaunch v c hasBias useSoftplus lower upper p memory).bind
      (runTrials v c hasBias useSoftplus lower upper rest)

theorem trials_frame (v : Views) (c : Whole.Host) (fresh : FreshPhysical v preexisting)
    (hasBias useSoftplus : Bool) (lower upper : α)
    (trials : List (LaunchPlan α c)) (memory : PhysicalMemory α) :
    ∃ final, runTrials v c hasBias useSoftplus lower upper trials memory = some final ∧
      (∀ allocation off, ¬ v.written c allocation off → final allocation off = memory allocation off) ∧
      (∀ allocation off, preexisting allocation → final allocation off = memory allocation off) := by
  induction trials generalizing memory with
  | nil => exact ⟨memory,rfl,fun _ _ _ => rfl,fun _ _ _ => rfl⟩
  | cons p rest ih =>
    obtain ⟨middle,ran,_,frame⟩ := physical_correct p.model (reblock c p.block) p.scan v
      hasBias useSoftplus lower upper p.order p.complete memory
    obtain ⟨final,tail,outside,old⟩ := ih middle
    have mf : ∀ allocation off, preexisting allocation → middle allocation off = memory allocation off :=
      fun allocation off h => frame allocation off (fresh_outside v (reblock c p.block) fresh allocation off h)
    refine ⟨final,?_,fun allocation off h => (outside allocation off h).trans (frame allocation off h),
      fun allocation off h => (old allocation off h).trans (mf allocation off h)⟩
    simp only [runTrials,runLaunch,ran,Option.bind_some,tail]

def tunedRun (v : Views) (c : Whole.Host) (hasBias useSoftplus : Bool) (lower upper : α)
    (trials : List (LaunchPlan α c)) (selected : LaunchPlan α c) (memory : PhysicalMemory α) :=
  (runTrials v c hasBias useSoftplus lower upper trials memory).bind
    (runLaunch v c hasBias useSoftplus lower upper selected)

theorem tuned_correct (v : Views) (c : Whole.Host) (fresh : FreshPhysical v preexisting)
    (hasBias useSoftplus : Bool) (lower upper : α)
    (trials : List (LaunchPlan α c)) (selected : LaunchPlan α c) (memory : PhysicalMemory α) :
    ∃ final, tunedRun v c hasBias useSoftplus lower upper trials selected memory = some final ∧
      PhysicalValues selected.model (reblock c selected.block) selected.scan v (v.words memory)
        hasBias useSoftplus lower upper final ∧
      (∀ allocation off, ¬ v.written c allocation off → final allocation off = memory allocation off) ∧
      (∀ allocation off, preexisting allocation → final allocation off = memory allocation off) := by
  obtain ⟨middle,tried,trialFrame,old⟩ := trials_frame v c fresh hasBias useSoftplus lower upper trials memory
  obtain ⟨final,ran,values,frame⟩ := physical_correct selected.model (reblock c selected.block)
    selected.scan v hasBias useSoftplus lower upper selected.order selected.complete middle
  rw [words_frame v fresh memory middle old] at values
  refine ⟨final,?_,values,
    fun allocation off h => (frame allocation off h).trans (trialFrame allocation off h),
    fun allocation off h => (frame allocation off (fresh_outside v _ fresh allocation off h)).trans (old allocation off h)⟩
  simp only [tunedRun,tried,Option.bind_some,runLaunch,ran]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBoundary
