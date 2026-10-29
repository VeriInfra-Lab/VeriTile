import bench.repairs.tritonbench_g.spinning_lock_reduction.Formula
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.KernelStages

/-! Finite logical stage outputs from the existing complete source execution.
Publication belongs only to producer programs. The exact trace frames retain
every other partial/lock cell and the independent output rectangle retains C.
-/
namespace VeriTile.Bench.Repairs.SpinningLock.FiniteResults
open VeriTile.Bench.TritonBenchG.SpinningLock
open SourceReader
open Classical
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 12000

def ProducerControl (h : HostArgs) (s : Wide.State V) : Prop :=
  ∀ p, match (s.pc p).phase with
    | .publish _ | .release => Wide.reducer h p = false
    | _ => True

theorem step_producerControl (b : Backend V) (h : HostArgs) (s : Wide.State V)
    (p : Nat) (valid : ProducerControl h s) : ProducerControl h (Wide.step b h p s).2 := by
  intro q
  by_cases same : q=p
  · subst q
    have old := valid p
    by_cases outside : p ≥ h.numSms
    · simpa [Wide.step,Wide.decode,outside,put] using old
    · cases phase : (s.pc p).phase <;>
        simp only [Wide.step,Wide.decode,if_neg outside,phase,put_same]
      all_goals try split_ifs
      all_goals simp_all [ProducerControl]
  · simpa [Wide.step,put,same] using valid q

theorem decode_publish (h : HostArgs) (s : Wide.State V) (p q lane : Nat) (a : Int)
    (valid : ProducerControl h s) (event : (Wide.decode h p s).1=.publish q lane a) :
    q=p ∧ lane<Wide.lanes h ∧ a=Wide.pAddress h p lane ∧ Wide.reducer h p=false := by
  have old := valid p
  by_cases outside : p ≥ h.numSms
  · simp [Wide.decode,outside] at event
  · cases phase : (s.pc p).phase <;>
      simp only [Wide.decode,if_neg outside,phase] at event
    all_goals try split_ifs at event
    all_goals simp_all [ProducerControl]
    all_goals (rcases event with ⟨rfl,rfl,rfl⟩; rfl)

theorem decode_release (h : HostArgs) (s : Wide.State V) (p : Nat) (a : Int)
    (valid : ProducerControl h s) (event : (Wide.decode h p s).1=.release a) :
    a=(p:Int) ∧ Wide.reducer h p=false := by
  have old := valid p
  by_cases outside : p ≥ h.numSms
  · simp [Wide.decode,outside] at event
  · cases phase : (s.pc p).phase <;>
      simp only [Wide.decode,if_neg outside,phase] at event
    all_goals try split_ifs at event
    all_goals simp_all [ProducerControl]

theorem publish_origin (b : Backend V) (h : HostArgs) (xs : List Nat)
    (s : Wide.State V) (valid : ProducerControl h s) (q lane : Nat) (a : Int)
    (event : Event.publish q lane a ∈ (Wide.run b h xs s).1) :
    q∈xs ∧ lane<Wide.lanes h ∧ a=Wide.pAddress h q lane ∧ Wide.reducer h q=false := by
  induction xs generalizing s with
  | nil => simp [Wide.run] at event
  | cons p rest ih =>
    rcases List.mem_cons.mp event with first|tail
    · obtain ⟨rfl,hl,ha,hr⟩ := decode_publish h s p q lane a valid first.symm
      exact ⟨by simp,hl,ha,hr⟩
    · obtain ⟨member,hl,ha,hr⟩ := ih (Wide.step b h p s).2
        (step_producerControl b h s p valid) tail
      exact ⟨List.mem_cons_of_mem _ member,hl,ha,hr⟩

theorem release_origin (b : Backend V) (h : HostArgs) (xs : List Nat)
    (s : Wide.State V) (valid : ProducerControl h s) (a : Int)
    (event : Event.release a ∈ (Wide.run b h xs s).1) :
    ∃ p∈xs,a=(p:Int) ∧ Wide.reducer h p=false := by
  induction xs generalizing s with
  | nil => simp [Wide.run] at event
  | cons p rest ih =>
    rcases List.mem_cons.mp event with first|tail
    · obtain ⟨ha,hr⟩ := decode_release h s p a valid first.symm
      exact ⟨p,by simp,ha,hr⟩
    · obtain ⟨q,member,ha,hr⟩ := ih (Wide.step b h p s).2
        (step_producerControl b h s p valid) tail
      exact ⟨q,List.mem_cons_of_mem _ member,ha,hr⟩

theorem projection (b : Backend V) (h : HostArgs) (buf : Wide.Buffers h)
    (stage : Phased.Stage) (xs : List Nat) (mem : Memory V)
    (result : List SourceEvent × Wide.State V)
    (facts : KernelProof.Facts b h buf stage xs mem result) :
    (result.1.map project,result.2)=
      Wide.run b h (Phased.activeSchedule h stage xs) (Wide.initial b mem) := by
  obtain ⟨canonical,ran,projected⟩ := KernelProof.source_run b h buf stage xs mem
  have same : result=canonical := Option.some.inj (facts.execution.symm.trans ran)
  simpa [same] using projected

theorem selected_producer (h : HostArgs) (stage : Phased.Stage) (xs : List Nat) (p : Nat)
    (member : p∈Phased.activeSchedule h stage xs) (role : Wide.reducer h p=false) :
    stage=.producers := by
  have selected := (List.mem_filter.mp member).2
  cases stage <;> simp_all [Phased.selected]

theorem output (b : Backend V) (h : HostArgs) (buf : Wide.Buffers h)
    (stage : Phased.Stage) (xs : List Nat) (mem : Memory V)
    (result : List SourceEvent × Wide.State V)
    (facts : KernelProof.Facts b h buf stage xs mem result) (i : Nat) :
    result.2.values.mem.output (i:Int)=
      if Formula.writes h stage i then b.zero else mem.output (i:Int) := by
  rw [facts.output]
  have same : KernelProof.Writes h stage (i:Int) ↔ Formula.writes h stage i := by
    cases stage <;> simp only [KernelProof.Writes,KernelProof.Role,Formula.writes,Formula.role,
      Int.ofNat_inj]
  simp only [KernelProof.outputTarget,same]

theorem partials (b : Backend V) (h : HostArgs) (buf : Wide.Buffers h)
    (stage : Phased.Stage) (xs : List Nat) (mem : Memory V)
    (result : List SourceEvent × Wide.State V)
    (facts : KernelProof.Facts b h buf stage xs mem result) (i : Nat)
    (inside : i<Formula.partialCount h) :
    result.2.values.mem.partials (i:Int)=
      if stage=.producers ∧ (i/(h.bm*h.bn))%h.groupK≠0 then b.zero else mem.partials (i:Int) := by
  have positive : 0<h.bm*h.bn := Nat.mul_pos buf.tileMPositive buf.tileNPositive
  by_cases selected : stage=.producers ∧ (i/(h.bm*h.bn))%h.groupK≠0
  · rw [if_pos selected]
    rcases selected with ⟨rfl,role⟩
    have pidBound : i/(h.bm*h.bn)<h.numSms :=
      (Nat.div_lt_iff_lt_mul positive).mpr (by simpa [Formula.partialCount,Nat.mul_assoc] using inside)
    have laneBound := Nat.mod_lt i positive
    have value := (KernelProof.producer_supplies_consumer b h buf xs mem result facts).2.2
      (i/(h.bm*h.bn)) pidBound role (i%(h.bm*h.bn)) laneBound
    have location : i/(h.bm*h.bn)*h.bm*h.bn+i%(h.bm*h.bn)=i := by
      calc
        _ = i%(h.bm*h.bn)+(h.bm*h.bn)*(i/(h.bm*h.bn)) := by
          rw [Nat.mul_assoc,Nat.mul_comm (i/(h.bm*h.bn)) (h.bm*h.bn),Nat.add_comm]
        _ = i := Nat.mod_add_div i (h.bm*h.bn)
    simpa only [location] using value
  · rw [if_neg selected]
    apply facts.partialFrame
    intro p lane member
    have projected := congrArg Prod.fst (projection b h buf stage xs mem result facts)
    change result.1.map project = _ at projected
    rw [projected] at member
    obtain ⟨listed,hl,ha,role⟩ := publish_origin b h _ _ (by intro p; trivial) p lane (i:Int) member
    have producer := selected_producer h stage xs p listed role
    have address : i=p*h.bm*h.bn+lane := Int.ofNat_inj.mp ha
    change lane < h.bm*h.bn at hl
    have quotient : i/(h.bm*h.bn)=p := by
      rw [address,Nat.mul_assoc,Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hl]
      simp
    apply selected
    exact ⟨producer,by simpa [quotient,Wide.reducer] using role⟩

theorem locks (b : Backend V) (h : HostArgs) (buf : Wide.Buffers h)
    (stage : Phased.Stage) (xs : List Nat) (mem : Memory V)
    (result : List SourceEvent × Wide.State V)
    (facts : KernelProof.Facts b h buf stage xs mem result) (p : Nat) (inside : p<h.numSms) :
    result.2.values.mem.locks (p:Int)=
      if stage=.producers ∧ p%h.groupK≠0 then 1 else mem.locks (p:Int) := by
  by_cases selected : stage=.producers ∧ p%h.groupK≠0
  · rw [if_pos selected]
    rcases selected with ⟨rfl,role⟩
    exact (KernelProof.producer_supplies_consumer b h buf xs mem result facts).1 p inside role
  · rw [if_neg selected]
    apply facts.lockFrame
    intro member
    have projected := congrArg Prod.fst (projection b h buf stage xs mem result facts)
    change result.1.map project = _ at projected
    rw [projected] at member
    obtain ⟨q,listed,ha,role⟩ := release_origin b h _ _ (by intro p; trivial) (p:Int) member
    have same : p=q := Int.ofNat_inj.mp ha
    subst q
    exact selected ⟨selected_producer h stage xs p listed role,by simpa [Wide.reducer] using role⟩

/-- The finite producer result retains the publication invariant for every
lock, including reducers whose partials and locks are not modified. -/
theorem producers_published (zero : V) (h : HostArgs)
    (input : Formula.Values V h outputCount)
    (published : Formula.Published zero h input) :
    Formula.Published zero h (Formula.stage zero h .producers input) := by
  intro p ready lane hl
  have positive : 0 < h.bm*h.bn := Nat.zero_lt_of_lt hl
  have quotient : (p.val*h.bm*h.bn+lane)/(h.bm*h.bn)=p.val := by
    rw [Nat.mul_assoc,Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hl]
    simp
  by_cases role : p.val%h.groupK≠0
  · simp [Formula.stage,quotient,role]
  · have equal : p.val%h.groupK=0 := Classical.not_not.mp role
    have initialReady : input.locks p=1 := by simpa [Formula.stage,equal] using ready
    simpa [Formula.stage,quotient,equal] using published p initialReady lane hl

theorem producer_to_consumer_domain (zero : V) (h : HostArgs)
    (input : Formula.Values V h outputCount)
    (domain : Formula.Domain zero h .producers input) :
    Formula.Domain zero h .consumers (Formula.stage zero h .producers input) :=
  ⟨producers_published zero h input domain.1,
    fun _ => Formula.producers_ready zero h input⟩

#axiomsClean producers_published
#axiomsClean producer_to_consumer_domain
#axiomsClean partials
#axiomsClean locks
#axiomsClean output
end
end VeriTile.Bench.Repairs.SpinningLock.FiniteResults
