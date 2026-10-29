import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.Contract
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.PhasedFPAdapter

noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.Admission
open VeriTile.Bench.TritonBenchG.SpinningLock VeriTile.Triton
open Classical

/-- Setup includes metadata queries and exact readiness readback. A setup
exception or guard rejection precedes both submissions. Query/submission failures
retain their actual accepted prefix. Both phases submit inside P.device's
standard device context on its same current stream; leaving the context restores
the ambient caller device, including submission exceptions.
Unit is the model's Python None token. -/
def wrapperHost (s : Scalars) (p c l : Tensor) (setup pq cq lq first second : Bool) : Phased.HostResult :=
  Phased.setupThenCall (setup && check s p c l)
    (if pq then .ok false else .error ())
    (if cq then .ok false else .error ())
    (if lq then .ok false else .error ()) first second

def rawFP (p c l : Tensor) : Memory (FP.Value .fp32) := memory (fun w => ⟨w⟩) p c l

/-- Execute the exact native-width graph of the two source invocations. The
independent target never defines this result. Only caller objects are mutated. -/
def rawResult (b : Backend V) (s : Scalars) (plan : Phased.Plan) (mem : Memory V) : Wide.WrapperResult V :=
  Wide.launch b s.args (Phased.flatten s.args plan) mem

structure Observation (V : Type) where
  partialObject : Nat
  outputObject : Nat
  lockObject : Nat
  host : Phased.HostResult
  device : Wide.WrapperResult V

def observe (b : Backend V) (s : Scalars) (p c l : Tensor)
    (setup pq cq lq first second : Bool) (plan : Phased.Plan) (mem : Memory V) : Observation V :=
  ⟨p.object,c.object,l.object,wrapperHost s p c l setup pq cq lq first second,
    rawResult b s plan mem⟩

theorem checked_none (ok : check s p c l = Bool.true) :
    (wrapperHost s p c l true true true true true true).returned=.ok () := by
  simp [wrapperHost,ok,Phased.setupThenCall,Phased.hostCall]

theorem rejected_no_submissions (bad : check s p c l = Bool.false) :
    (wrapperHost s p c l setup pq cq lq first second).submitted=[] := by
  simp [wrapperHost,bad,Phased.setupThenCall]

theorem checked_prefix (b : Backend V) (addzero : b.add b.zero b.zero=b.zero)
    (decode : BitVec 32 → V) (zero : decode 0=b.zero) (ok : check s p c l = Bool.true)
    (plan : Phased.Plan) :
    Wide.PrefixFacts b s.args (checkedBuffers ok) (Phased.flatten s.args plan)
      (memory decode p c l) (rawResult b s plan (memory decode p c l)) :=
  Wide.ready_prefix_backend b addzero s.args (checkedBuffers ok) _ _ (checked_ready b decode zero ok)

theorem checked_synchronized (b : Backend V) (addzero : b.add b.zero b.zero=b.zero)
    (decode : BitVec 32 → V) (zero : decode 0=b.zero) (ok : check s p c l = Bool.true)
    (plan : Phased.Plan) (sync : Phased.Synchronized b s.args plan (memory decode p c l)) :
    Wide.CompleteFacts b s.args (checkedBuffers ok) (Phased.flatten s.args plan)
      (memory decode p c l) (rawResult b s plan (memory decode p c l)) := by
  refine ⟨checked_prefix b addzero decode zero ok plan,?_⟩
  rw [rawResult,Wide.launch_eq_run b s.args (checkedBuffers ok)]
  exact sync.all_done

theorem no_submission_unchanged (b : Backend V)
    (host : Phased.HostResult) (plan : Phased.Plan) (mem : Memory V)
    (empty : host.submitted=[]) (allowed : Phased.AllowedExecution b s.args host plan mem) :
    (rawResult b s plan mem).events=[] ∧ (rawResult b s plan mem).final.values.mem=mem := by
  have hp : plan.producers=[] := by
    by_contra hn
    have hh := allowed.producerSubmitted hn
    simp [empty] at hh
  have hc : plan.consumers=[] := by
    by_contra hn
    have hh := allowed.consumerSubmitted hn
    simp [empty] at hh
  simp [rawResult,Phased.flatten,hp,hc,Phased.activeSchedule,
    Wide.launch,Wide.nativeRun,Wide.initial,VeriTile.Bench.TritonBenchG.SpinningLock.initial]

theorem rejected_unchanged (b : Backend V) (bad : check s p c l = Bool.false)
    (plan : Phased.Plan) (mem : Memory V)
    (allowed : Phased.AllowedExecution b s.args (wrapperHost s p c l setup pq cq lq first second) plan mem) :
    (rawResult b s plan mem).events=[] ∧ (rawResult b s plan mem).final.values.mem=mem :=
  no_submission_unchanged b _ plan mem (rejected_no_submissions bad) allowed

abbrev ByteHeap := Nat → BitVec 8

def byteOf (w : BitVec 32) (j : Nat) : BitVec 8 := BitVec.ofNat 8 (w.toNat / 2^(8*j))
def contains (t : Tensor) (a : Nat) : Prop := t.base≤a ∧ a<t.base+4*t.elements

def slot (t : Tensor) (a : Nat) : Int := ((a-t.base)/4 : Nat)

/-- Ordinary contiguous, little-endian four-byte tensor representation. These
are per-byte input observations, not a premise about any computed output. -/
structure InputBytes (before : ByteHeap) (p c l : Tensor) : Prop where
  partials : ∀ a, contains p a → before a=byteOf (p.word ((a-p.base)/4)) ((a-p.base)%4)
  output : ∀ a, contains c a → before a=byteOf (c.word ((a-c.base)/4)) ((a-c.base)%4)
  locks : ∀ a, contains l a → before a=byteOf (l.word ((a-l.base)/4)) ((a-l.base)%4)

/-- Extensional byte image of the source word memory under the ordinary
load/store codec. Bytes outside the three caller windows use the original heap. -/
def heap (before : ByteHeap) (p c l : Tensor) (mem : Memory (FP.Value .fp32)) : ByteHeap := fun a =>
  if contains p a then byteOf (mem.partials (slot p a)).bits ((a-p.base)%4)
  else if contains c a then byteOf (mem.output (slot c a)).bits ((a-c.base)%4)
  else if contains l a then byteOf (mem.locks (slot l a)) ((a-l.base)%4)
  else before a

def Untouched (events : List Event) (p c l : Tensor) (a : Nat) : Prop :=
  (contains p a → ∀ pid lane, Event.publish pid lane (slot p a) ∉ events) ∧
  (contains c a → ∀ pid lane, Event.output pid lane (slot c a) ∉ events) ∧
  (contains l a → Event.release (slot l a) ∉ events)

theorem complete_byte_frame (ok : check s p c l = Bool.true) (plan : Phased.Plan)
    (before : ByteHeap) (input : InputBytes before p c l) (a : Nat)
    (untouched : Untouched (rawResult fp32 s plan (rawFP p c l)).events p c l a) :
    heap before p c l (rawResult fp32 s plan (rawFP p c l)).final.values.mem a=before a := by
  have facts := checked_prefix fp32 fp32_positive_zero_add (fun w => (⟨w⟩ : FP.Value .fp32)) rfl ok plan
  unfold rawFP at *
  unfold heap
  split_ifs with hp hc hl
  · rw [facts.partialFrame _ (untouched.1 hp)]
    simpa [memory,slot] using (input.partials a hp).symm
  · rw [facts.outputFrame _ (untouched.2.1 hc)]
    simpa [memory,slot] using (input.output a hc).symm
  · rw [facts.lockFrame _ (untouched.2.2 hl)]
    have hslot : (a-l.base)/4<l.elements := by
      have hh := hl.2
      have hb := hl.1
      apply (Nat.div_lt_iff_lt_mul (by omega : 0<4)).mpr
      omega
    simpa only [memory,slot,Int.toNat_natCast,Int.natCast_nonneg,true_and,if_pos hslot] using (input.locks a hl).symm
  · rfl

theorem heap_output_bytes (ok : check s p c l = Bool.true) (before : ByteHeap)
    (mem : Memory (FP.Value .fp32)) (i j : Nat) (hi : i<c.elements) (hj : j<4) :
    heap before p c l mem (c.base+4*i+j)=byteOf (mem.output (i:Nat)).bits j := by
  have facts := (checked_parts ok).2.2.2.2.1
  rcases facts with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,dpc,_,_,_,_⟩
  have hc : contains c (c.base+4*i+j) := by constructor <;> omega
  have hp : ¬contains p (c.base+4*i+j) := by
    intro hp
    rcases hp with ⟨hp₁,hp₂⟩
    rcases dpc with hd|hd <;> omega
  have hd : c.base+4*i+j-c.base=4*i+j := by omega
  have hdiv : (4*i+j)/4=i := by omega
  have hmod : (4*i+j)%4=j := by omega
  simp only [heap,if_neg hp,if_pos hc,slot,hd,hdiv,hmod]

#axiomsClean checked_none
#axiomsClean checked_prefix
#axiomsClean checked_synchronized
#axiomsClean no_submission_unchanged
#axiomsClean rejected_unchanged
#axiomsClean complete_byte_frame
#axiomsClean heap_output_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Admission
