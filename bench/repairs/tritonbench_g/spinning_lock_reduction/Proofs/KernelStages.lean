import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.SourceBridge
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.PhasedTermination

/-! Kernel-only stage results. The target is an independent tile/rectangle
formula. Producer completion is constructed. Consumer completion requires the
published producer input, which is obtained from every completed producer run.
No wrapper submission, stream implementation or output equation is assumed.
-/
noncomputable section
open Classical
namespace VeriTile.Bench.Repairs.SpinningLock.KernelProof
open VeriTile.Bench.TritonBenchG.SpinningLock
open SourceReader

def Role (h : HostArgs) (stage : Phased.Stage) (p : Nat) : Prop :=
  match stage with
  | .producers => p % h.groupK ≠ 0
  | .consumers => p % h.groupK = 0

/-- Independent source-stage output rectangle, including both tile tails. -/
def Writes (h : HostArgs) (stage : Phased.Stage) (a : Int) : Prop :=
  ∃ p < h.numSms, Role h stage p ∧
    ∃ row < h.bm, ∃ col < h.bn,
      row < h.m ∧ p * h.bn + col < h.n ∧
      ((row * h.strideM + (p * h.bn + col) * h.strideN : Nat) : Int) = a

def outputTarget (b : Backend V) (h : HostArgs) (stage : Phased.Stage)
    (before : Memory V) (a : Int) : V :=
  if Writes h stage a then b.zero else before.output a

/-- The consumer's extra stage input is readiness, not an execution/result law.
The published-zero condition is already required by the unchanged guards. -/
def ConsumerReady (h : HostArgs) (mem : Memory V) : Prop :=
  ∀ p < h.numSms, p % h.groupK ≠ 0 → mem.locks (p : Int) = 1

theorem role_selected (h : HostArgs) (stage : Phased.Stage) (p : Nat) :
    Role h stage p ↔ Phased.selected h stage p = Bool.true := by
  cases stage <;> simp [Role, Phased.selected, Wide.reducer]

theorem writes_iff (h : HostArgs) (buf : Wide.Buffers h) (stage : Phased.Stage) (a : Int) :
    Writes h stage a ↔ ∃ p < h.numSms, Phased.selected h stage p = Bool.true ∧
      ∃ lane < Wide.lanes h, Wide.mask h p lane = Bool.true ∧ Wide.cAddress h p lane = a := by
  constructor
  · rintro ⟨p,hp,role,row,hr,col,hc,hm,hn,ha⟩
    let lane := row * h.bn + col
    have hdiv : lane / h.bn = row := by
      dsimp [lane]
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ buf.tileNPositive,
        Nat.div_eq_of_lt hc, Nat.zero_add]
    have hmod : lane % h.bn = col := by simp [lane, Nat.add_mod, Nat.mod_eq_of_lt hc]
    have hbound : lane < Wide.lanes h := by
      have bound := Nat.mul_le_mul_right h.bn (show row+1 ≤ h.bm by omega)
      simp only [Nat.add_mul, Nat.one_mul] at bound
      dsimp [lane, Wide.lanes]
      omega
    refine ⟨p,hp,(role_selected h stage p).mp role,lane,hbound,?_,?_⟩
    · simp [Wide.mask,hdiv,hmod,hm,hn]
    · simpa [Wide.cAddress,hdiv,hmod] using ha
  · rintro ⟨p,hp,selected,lane,hl,mask,ha⟩
    have hm : lane / h.bn < h.m ∧ p*h.bn+lane%h.bn<h.n := by simpa [Wide.mask] using mask
    exact ⟨p,hp,(role_selected h stage p).mpr selected,
      lane/h.bn,(Nat.div_lt_iff_lt_mul buf.tileNPositive).mpr hl,
      lane%h.bn,Nat.mod_lt _ buf.tileNPositive,hm.1,hm.2,ha⟩

theorem decode_output_pid (h : HostArgs) (s : Wide.State V) (p q lane : Nat) (a : Int)
    (he : (Wide.decode h p s).1 = .output q lane a) : q = p := by
  by_cases hg : p ≥ h.numSms
  · simp [Wide.decode,hg] at he
  · cases hp : (s.pc p).phase <;> simp only [Wide.decode,if_neg hg,hp] at he
    all_goals try split_ifs at he
    all_goals simp_all

theorem output_pid_mem (b : Backend V) (h : HostArgs) (xs : List Nat)
    (s : Wide.State V) (q lane : Nat) (a : Int)
    (he : Event.output q lane a ∈ (Wide.run b h xs s).1) : q ∈ xs := by
  induction xs generalizing s with
  | nil => simp [Wide.run] at he
  | cons p ps ih =>
    rcases List.mem_cons.mp he with he | he
    · exact List.mem_cons.mpr (Or.inl (decode_output_pid h s p q lane a he.symm))
    · exact List.mem_cons_of_mem _ (ih _ he)

theorem completed_writes (b : Backend V) (h : HostArgs) (buf : Wide.Buffers h)
    (stage : Phased.Stage) (xs : List Nat) (mem : Memory V)
    (done : Phased.StageDone h stage
      (Wide.run b h (Phased.activeSchedule h stage xs) (Wide.initial b mem)).2) (a : Int) :
    Visited (Wide.run b h (Phased.activeSchedule h stage xs) (Wide.initial b mem)).1 a ↔
      Writes h stage a := by
  rw [writes_iff h buf stage a]
  constructor
  · rintro ⟨p,lane,he⟩
    obtain ⟨hp,hl,hm,ha⟩ := Wide.run_output_origin h b _ _ p lane a he
    have selected := (List.mem_filter.mp (output_pid_mem b h _ _ p lane a he)).2
    have hs : Phased.selected h stage p = Bool.true := by
      simpa [hp] using selected
    exact ⟨p,hp,hs,lane,hl,hm,ha⟩
  · rintro ⟨p,hp,selected,lane,hl,hm,rfl⟩
    exact ⟨p,lane,Wide.completed_lane_visited b h _ mem p lane hp hl hm (done p hp selected)⟩

theorem stage_target (b : Backend V) (addzero : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (buf : Wide.Buffers h) (stage : Phased.Stage) (xs : List Nat)
    (mem : Memory V) (ready : Wide.PublishedReady b h mem)
    (done : Phased.StageDone h stage
      (Wide.run b h (Phased.activeSchedule h stage xs) (Wide.initial b mem)).2) :
    ∀ a, (Wide.run b h (Phased.activeSchedule h stage xs) (Wide.initial b mem)).2.values.mem.output a =
      outputTarget b h stage mem a := by
  intro a
  rw [Wide.run_zero_output b addzero h _ _ (Wide.initial_ready b h mem ready),
    completed_writes b h buf stage xs mem done a]
  rfl

theorem source_run (b : Backend V) (h : HostArgs) (buf : Wide.Buffers h)
    (stage : Phased.Stage) (xs : List Nat) (mem : Memory V) :
    ∃ result, runKernel b h stage xs mem = some result ∧
      (result.1.map project,result.2) =
        Wide.run b h (Phased.activeSchedule h stage xs) (Wide.initial b mem) := by
  refine ⟨runPlan b sourcePlan h stage xs (Wide.initial b mem),?_,?_⟩
  · simp [runKernel,compileAndRun,source_compiles,source_assertion,initial_projection]
  · rw [run_projection]
    exact Phased.stageRun_eq b h buf stage xs _ (by intro p; trivial)

/-- Full result at the source boundary: successful decoding/execution, complete
stage output, every memory frame, and the source-order value reference. -/
structure Facts (b : Backend V) (h : HostArgs) (buf : Wide.Buffers h)
    (stage : Phased.Stage) (xs : List Nat) (mem : Memory V)
    (result : List SourceEvent × Wide.State V) : Prop where
  execution : runKernel b h stage xs mem = some result
  done : Phased.StageDone h stage result.2
  output : ∀ a, result.2.values.mem.output a = outputTarget b h stage mem a
  invariant : Wide.ZeroInvariant b h result.2
  bounded : ∀ e ∈ result.1.map project, Wide.EventInBounds buf e
  reference : Related b result.2.values (replay b (result.1.map project) (Reference.initial mem))
  partialFrame : ∀ a, (∀ p l, Event.publish p l a ∉ result.1.map project) → result.2.values.mem.partials a = mem.partials a
  outputFrame : ∀ a, (∀ p l, Event.output p l a ∉ result.1.map project) → result.2.values.mem.output a = mem.output a
  lockFrame : ∀ a, Event.release a ∉ result.1.map project → result.2.values.mem.locks a = mem.locks a

theorem facts_of_done (b : Backend V) (addzero : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (buf : Wide.Buffers h) (stage : Phased.Stage) (xs : List Nat)
    (mem : Memory V) (ready : Wide.PublishedReady b h mem)
    (done : Phased.StageDone h stage
      (Wide.run b h (Phased.activeSchedule h stage xs) (Wide.initial b mem)).2) :
    ∃ result, Facts b h buf stage xs mem result := by
  obtain ⟨result,exec,projection⟩ := source_run b h buf stage xs mem
  have eqs := Prod.mk.inj projection
  refine ⟨result,exec,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · simpa [eqs.2] using done
  · simpa [eqs.2] using stage_target b addzero h buf stage xs mem ready done
  · rw [eqs.2]
    exact Wide.run_zeroInvariant b addzero h _ _ (Wide.initial_ready b h mem ready)
  · rw [eqs.1]
    exact Wide.run_in_bounds h buf b _ _ (by intro p; trivial)
  · rw [eqs.1,eqs.2]
    exact Wide.run_related b h _ _ _ (related_initial b mem)
  · rw [eqs.1,eqs.2]
    exact fun a => (Wide.full_schedule_frame b h _ mem a).1
  · rw [eqs.1,eqs.2]
    exact fun a => (Wide.full_schedule_frame b h _ mem a).2.1
  · rw [eqs.1,eqs.2]
    exact fun a => (Wide.full_schedule_frame b h _ mem a).2.2

theorem producer_complete (b : Backend V) (addzero : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (buf : Wide.Buffers h) (mem : Memory V)
    (ready : Wide.PublishedReady b h mem) :
    ∃ xs result, Facts b h buf .producers xs mem result := by
  obtain ⟨s,⟨xs,hxs⟩,_,done,_⟩ := Phased.complete_producers b h buf mem
  have done' : Phased.StageDone h .producers
      (Wide.run b h (Phased.activeSchedule h .producers xs) (Wide.initial b mem)).2 := by
    rw [hxs]
    intro p hp hs
    exact done p hp (by simpa [Phased.selected] using hs)
  obtain ⟨result,facts⟩ := facts_of_done b addzero h buf .producers xs mem ready done'
  exact ⟨xs,result,facts⟩

theorem producer_supplies_consumer (b : Backend V) (h : HostArgs) (buf : Wide.Buffers h)
    (xs : List Nat) (mem : Memory V) (result : List SourceEvent × Wide.State V)
    (facts : Facts b h buf .producers xs mem result) :
    ConsumerReady h result.2.values.mem ∧ Wide.PublishedReady b h result.2.values.mem ∧
      ∀ p < h.numSms, p % h.groupK ≠ 0 → ∀ lane < h.bm*h.bn,
        result.2.values.mem.partials ((p*h.bm*h.bn+lane : Nat) : Int) = b.zero := by
  obtain ⟨canonical,hex,projection⟩ := source_run b h buf .producers xs mem
  have he : result = canonical := Option.some.inj (facts.execution.symm.trans hex)
  have eqs := Prod.mk.inj projection
  have done := facts.done
  rw [he,eqs.2] at done
  have ready := Phased.every_completed_producer_stage_ready b h xs mem done
  have ready' : ConsumerReady h result.2.values.mem := by
    intro p hp role
    rw [he,eqs.2]
    exact ready p hp (by simp [Wide.reducer,role])
  refine ⟨ready',facts.invariant.ready,?_⟩
  intro p hp role lane hl
  exact facts.invariant.ready p (ready' p hp role) lane hl

theorem consumers_prefix (b : Backend V) (h : HostArgs) (buf : Wide.Buffers h)
    (mem : Memory V) (ready : ConsumerReady h mem) (n : Nat) (hn : n ≤ h.numSms) :
    ∃ s, Phased.StageScheduled b h .consumers (Wide.initial b mem) s ∧
      Wide.ProducersReady h s ∧
      (∀ p < n, Wide.reducer h p = Bool.true → (s.pc p).phase = .done) ∧
      (∀ p < h.numSms, n ≤ p → s.pc p = ⟨1,.branch⟩) := by
  induction n with
  | zero =>
    refine ⟨Wide.initial b mem,Phased.StageScheduled.refl b h .consumers _,?_,by simp,by simp [Wide.initial]⟩
    intro p hp hr
    exact ready p hp (by simpa [Wide.reducer] using hr)
  | succ n ih =>
    obtain ⟨s,hs,sready,done,untouched⟩ := ih (by omega)
    by_cases role : Wide.reducer h n = Bool.true
    · obtain ⟨t,ht,htc,_⟩ := Wide.solo_initial_complete b h buf n s (by omega)
        (fun _ => sready) (untouched n (by omega) (by omega))
      refine ⟨t,hs.trans (Phased.solo_stage ht (by omega) role),ht.producersReady sready,?_,?_⟩
      · intro p hp hr
        by_cases he : p = n
        · subst p; exact htc
        · rw [ht.other_control he]
          exact done p (by omega) hr
      · intro p hp hn
        rw [ht.other_control (by omega)]
        exact untouched p hp (by omega)
    · refine ⟨s,hs,sready,?_,?_⟩
      · intro p hp hr
        have he : p ≠ n := by intro he; subst p; exact role hr
        exact done p (by omega) hr
      · intro p hp hn
        exact untouched p hp (by omega)

theorem consumer_complete (b : Backend V) (addzero : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (buf : Wide.Buffers h) (mem : Memory V)
    (published : Wide.PublishedReady b h mem) (ready : ConsumerReady h mem) :
    ∃ xs result, Facts b h buf .consumers xs mem result := by
  obtain ⟨s,⟨xs,hxs⟩,_,done,_⟩ := consumers_prefix b h buf mem ready h.numSms (by omega)
  have done' : Phased.StageDone h .consumers
      (Wide.run b h (Phased.activeSchedule h .consumers xs) (Wide.initial b mem)).2 := by
    rw [hxs]
    exact done
  obtain ⟨result,facts⟩ := facts_of_done b addzero h buf .consumers xs mem published done'
  exact ⟨xs,result,facts⟩

#specNonCircular Writes avoiding [runKernel, Wide.run, Wide.nativeRun, SourceReader.runPlan]
#specNonCircular ConsumerReady avoiding [runKernel, Wide.run, Wide.nativeRun]
#axiomsClean producer_complete
#axiomsClean producer_supplies_consumer
#axiomsClean consumer_complete
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.KernelProof
