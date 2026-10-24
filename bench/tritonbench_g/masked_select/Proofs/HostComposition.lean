import bench.tritonbench_g.masked_select.Proofs.NativeDomain

namespace VeriTile.Bench.TritonBenchG.MaskedSelect.Whole
open VeriTile.Triton FP.Structural TypedCopy StableSelection
noncomputable section
open Classical
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Distinct logical producer buffers; output is freshly allocated. Physical
read-buffer aliases are harmless because no kernel writes a read buffer. -/
def ProducerSeparate (p : Parameters) : Prop :=
  p.input ≠ p.selection.cast ∧ p.input ≠ p.prefixSum.cast ∧
    (p.selection.cast : RegionName) ≠ p.prefixSum.cast

/-- An ordinary Torch primitive may allocate owned scratch, or return an
already-contiguous buffer. In the latter case materialization is an identity on
existing memory. This contract never specifies final selected output cells. -/
structure ProducerOwnership {α : Type} (p : Parameters) (s : State α)
    (input : Nat → α) (mask : Nat → Bool) (owned : RegionName → Nat → Prop) : Prop where
  payload : ∀ i, i < p.size → ¬ owned p.input i → s.mem p.input i = .mk .real (input i)
  selection : ∀ i, i < p.size → ¬ owned p.selection.cast i →
    s.mem p.selection.cast i = .mk .bool (mask i)
  cumulative : ∀ i, i < p.size → ¬ owned p.prefixSum.cast i →
    s.mem p.prefixSum.cast i = .mk .nat (count mask (i + 1))

/-- Allocation ownership is bounded by the three actual producer windows and
disjoint from all cells that existed before the call. Fresh output is separate.
Thus choosing `owned = all memory` cannot discharge the public frame theorem. -/
structure ScratchContract (p : Parameters) (length : Nat)
    (allocatedBefore owned : RegionName → Nat → Prop) : Prop where
  producerWindows : ∀ r i, owned r i →
    (r = p.input ∨ r = p.selection.cast ∨ r = p.prefixSum.cast) ∧ i < p.size
  scratchFresh : ∀ r i, owned r i → ¬ allocatedBefore r i
  outputFresh : ∀ i, i < length → ¬ allocatedBefore p.output i

theorem prepare_reads {α : Type} (p : Parameters) (s : State α)
    (input : Nat → α) (mask : Nat → Bool) (sep : ProducerSeparate p) :
    Reads p (prepare p input mask s) input mask := by
  constructor
  · intro i hi
    simp [prepare, materialize, sep.1, sep.2.1, hi]
  · intro i hi
    simp [prepare, materialize, sep.2.2, hi]
  · intro i hi
    simp [prepare, materialize, hi]

theorem prepare_frame {α : Type} (p : Parameters) (s : State α)
    (input : Nat → α) (mask : Nat → Bool) (owned : RegionName → Nat → Prop)
    (h : ProducerOwnership p s input mask owned) (r : RegionName) (o : Nat)
    (hr : ¬ owned r o) : (prepare p input mask s).mem r o = s.mem r o := by
  simp only [prepare, materialize]
  split
  · rename_i hcase
    rcases hcase with ⟨rfl, ho⟩
    exact (h.cumulative o ho hr).symm
  · split
    · rename_i hcase
      rcases hcase with ⟨rfl, ho⟩
      exact (h.selection o ho hr).symm
    · split
      · rename_i hcase
        rcases hcase with ⟨rfl, ho⟩
        exact (h.payload o ho hr).symm
      · rfl

def ValidPlan (p : Parameters) (plan : LaunchPlan) : Prop :=
  plan.config ∈ configs ∧ plan.jobs.Perm (List.range (grid p.size plan.config.width))

theorem valid_complete (p : Parameters) (plan : LaunchPlan) (h : ValidPlan p plan) :
    ∀ pid, pid < grid p.size plan.config.width → pid ∈ plan.jobs := by
  intro pid hp
  exact h.2.mem_iff.mpr (List.mem_range.mpr hp)

theorem plan_native (p : Parameters) (plan : LaunchPlan) (h : ValidPlan p plan)
    (hn : p.size < 2^31) :
    ∀ pid ∈ plan.jobs, ∀ lane, lane < plan.config.width →
      (index32 plan.config.width pid lane).toInt = (pid * plan.config.width + lane : Nat) ∧
      mask32 p.size plan.config.width pid lane = decide (pid * plan.config.width + lane < p.size) := by
  intro pid hp lane hl
  have hpid := List.mem_range.mp (h.2.mem_iff.mp hp)
  exact ⟨index32_exact plan.config h.1 p.size pid lane hn hpid hl,
    mask32_exact plan.config h.1 p.size pid lane hn hpid hl⟩

theorem launch_reads {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (plans : List LaunchPlan) (s : State α) (input : Nat → α) (mask : Nat → Bool)
    (h : Reads p s input mask) (sep : Separate p) :
    ∃ t, runLaunches rest p plans s = some t ∧ Reads p t input mask ∧
      (∀ r o, r ≠ p.output ∨ count mask p.size ≤ o → t.mem r o = s.mem r o) := by
  induction plans generalizing s with
  | nil => exact ⟨s, rfl, h, fun _ _ _ => rfl⟩
  | cons plan plans ih =>
    let q := parametersFor p plan
    have hs : Reads q s input mask := ⟨h.payload, h.selection, h.cumulative⟩
    have hsep : Separate q := sep
    have hg := grid_reads rest q plan.jobs s input mask hs hsep
    obtain ⟨t, ht, hr, hf⟩ := ih (gridValue rest q plan.jobs s)
      ⟨hg.payload, hg.selection, hg.cumulative⟩
    refine ⟨t, ?_, hr, ?_⟩
    · simp only [runLaunches, runGrid_value, bind, Option.bind]
      exact ht
    · intro r o ho
      rw [hf r o ho]
      exact grid_frame rest q plan.jobs s input mask hs hsep r o ho

/-- Primitive producer, varying-width autotune trials, selected source grid,
and actual returned tensor compose into the stable mathematical target. -/
theorem host_complete {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (input : Tensor α) (mask : Tensor Bool) (shape : List Nat)
    (trials : List LaunchPlan) (selected : LaunchPlan) (s : State α)
    (owned : RegionName → Nat → Prop)
    (broadcast : broadcastShape input.shape mask.shape = some shape)
    (sameDevice : input.device = mask.device) (nonempty : 0 < shape.prod)
    (producerSep : ProducerSeparate p) (outputSep : Separate p)
    (ownership : ProducerOwnership { p with size := shape.prod } s
      (broadcastValues input shape) (broadcastValues mask shape) owned)
    (selectedValid : ValidPlan { p with size := shape.prod } selected) :
    ∃ t, runHost rest p input mask trials selected s =
        .ok (returned p input.device (stableResult input mask shape).length, t) ∧
      (∀ k, k < (stableResult input mask shape).length →
        some (t.mem p.output k) = ((stableResult input mask shape)[k]?).map (Cell.mk .real)) ∧
      (∀ r o, ¬ owned r o → r ≠ p.output ∨ (stableResult input mask shape).length ≤ o →
        t.mem r o = s.mem r o) ∧
      (returned p input.device (stableResult input mask shape).length).device = mask.device := by
  let q : Parameters := { p with size := shape.prod }
  let values := broadcastValues input shape
  let choices := broadcastValues mask shape
  let produced := prepare q values choices s
  have hp := prepare_reads q s values choices producerSep
  have hlast : prefixLast q produced = count choices shape.prod := by
    unfold prefixLast
    rw [hp.cumulative (q.size - 1) (by change shape.prod - 1 < shape.prod; omega), Cell.read_same]
    congr 1
    change shape.prod - 1 + 1 = shape.prod
    omega
  obtain ⟨trialState, ht, hr, hf⟩ := launch_reads rest q trials produced values choices hp outputSep
  let chosen := parametersFor q selected
  obtain ⟨t, he, _, hout, hframe⟩ := complete_grid rest chosen selected.jobs trialState values choices
    ⟨hr.payload, hr.selection, hr.cumulative⟩ outputSep
    (config_width_pos selected.config selectedValid.1) (valid_complete q selected selectedValid)
  refine ⟨t, ?_, hout, ?_, sameDevice⟩
  · simp only [runHost, broadcast]
    rw [if_neg (by omega)]
    change (match runLaunches rest q trials produced with
      | none => Except.error HostFailure.kernelExecution
      | some trialState => match runGrid rest chosen selected.jobs trialState with
        | none => Except.error HostFailure.kernelExecution
        | some finalState => Except.ok (returned q input.device (prefixLast q produced), finalState)) = _
    simp only [ht]
    rw [he, hlast]
    simp [stableResult, returned, q, choices]
  · intro r o notOwned outside
    have outside' : r ≠ p.output ∨ count choices shape.prod ≤ o := by
      simpa [stableResult, choices] using outside
    rw [hframe r o outside', hf r o outside']
    exact prepare_frame q s values choices owned ownership r o notOwned

def NativePlanSafe (p : Parameters) (plan : LaunchPlan) : Prop :=
  ∀ pid ∈ plan.jobs, ∀ lane, lane < plan.config.width →
    (index32 plan.config.width pid lane).toInt = (pid * plan.config.width + lane : Nat) ∧
    mask32 p.size plan.config.width pid lane = decide (pid * plan.config.width + lane < p.size)

def NativeAccessSafe (mask : Nat → Bool) (n : Nat) (buffers : NativeBuffers n (count mask n)) : Prop :=
  ∀ i, i < n →
    (prefix64 mask i).toInt = (count mask (i + 1) : Nat) ∧
    (∀ byte, byte < 4 →
      (buffers.input.address i byte).toNat = buffers.input.base + i * 4 + byte) ∧
    (buffers.selection.address i 0).toNat = buffers.selection.base + i ∧
    (∀ byte, byte < 8 →
      (buffers.cumulative.address i byte).toNat = buffers.cumulative.base + i * 8 + byte) ∧
    (mask i = Bool.true → (destination64 mask i).toInt = (count mask i : Nat) ∧
      ∀ byte, byte < 4 →
        (BitVec.ofNat 64 buffers.output.base + destination64 mask i * 4 + BitVec.ofNat 64 byte).toNat =
          buffers.output.base + count mask i * 4 + byte)

theorem native_access_safe (mask : Nat → Bool) (n : Nat) (hn : n < 2^31)
    (buffers : NativeBuffers n (count mask n)) : NativeAccessSafe mask n buffers := by
  intro i hi
  refine ⟨prefix64_exact mask n i hn hi, ?_, ?_, ?_, ?_⟩
  · intro byte hb
    simpa [buffers.inputWidth] using allocation_address buffers.input i byte
      (by rw [buffers.inputLength]; exact hi) (by rw [buffers.inputWidth]; exact hb)
  · simpa [buffers.selectionWidth] using allocation_address buffers.selection i 0
      (by rw [buffers.selectionLength]; exact hi) (by rw [buffers.selectionWidth]; decide)
  · intro byte hb
    simpa [buffers.cumulativeWidth] using allocation_address buffers.cumulative i byte
      (by rw [buffers.cumulativeLength]; exact hi) (by rw [buffers.cumulativeWidth]; exact hb)
  · intro hm
    refine ⟨(destination64_selected mask n i hn hi hm).1, ?_⟩
    intro byte hb
    simpa [buffers.outputWidth] using destination_pointer mask n i hn hi hm buffers.output
      buffers.outputLength byte (by rw [buffers.outputWidth]; exact hb)

theorem existing_memory_preserved {α : Type} (p : Parameters) (length : Nat)
    (allocatedBefore owned : RegionName → Nat → Prop) (s t : State α)
    (scratch : ScratchContract p length allocatedBefore owned)
    (frame : ∀ r o, ¬ owned r o → r ≠ p.output ∨ length ≤ o → t.mem r o = s.mem r o) :
    ∀ r o, allocatedBefore r o → t.mem r o = s.mem r o := by
  intro r o old
  apply frame r o (fun h => scratch.scratchFresh r o h old)
  by_cases hr : r = p.output
  · right
    subst r
    by_contra ho
    exact scratch.outputFresh o (by omega) old
  · exact Or.inl hr

theorem host_shape_failure {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (input : Tensor α) (mask : Tensor Bool) (trials : List LaunchPlan)
    (selected : LaunchPlan) (s : State α)
    (h : broadcastShape input.shape mask.shape = none) :
    runHost rest p input mask trials selected s = .error .broadcastAssertion := by
  simp [runHost, h]

theorem host_empty_failure {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (input : Tensor α) (mask : Tensor Bool) (shape : List Nat) (trials : List LaunchPlan)
    (selected : LaunchPlan) (s : State α)
    (h : broadcastShape input.shape mask.shape = some shape) (empty : shape.prod = 0) :
    runHost rest p input mask trials selected s = .error .prefixLast := by
  simp [runHost, h, empty]

theorem all_false_output {α : Type} (input : Tensor α) (mask : Tensor Bool) (shape : List Nat)
    (h : ∀ i, i < shape.prod → broadcastValues mask shape i = Bool.false) :
    (stableResult input mask shape).length = 0 := by
  simp only [stableResult, length_select]
  exact all_false_count _ h

#axiomsClean host_complete
#axiomsClean host_shape_failure
#axiomsClean host_empty_failure
#axiomsClean all_false_output
end
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.MaskedSelect.Whole
