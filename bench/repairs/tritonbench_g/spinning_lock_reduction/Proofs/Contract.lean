import bench.repairs.tritonbench_g.spinning_lock_reduction.Contract
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.PhasedTermination

namespace VeriTile.Bench.Repairs.SpinningLock.Admission
open VeriTile.Bench.TritonBenchG.SpinningLock

theorem checked_parts (ok : check s p c l = Bool.true) :
    scalarCheck s = Bool.true ∧ tensorCheck p .fp32 p.device = Bool.true ∧
    tensorCheck c .fp32 p.device = Bool.true ∧ tensorCheck l .int32 p.device = Bool.true ∧
    Metadata s.args p c l ∧ readyCheck s.args p l = Bool.true := by
  simpa only [check,Bool.and_eq_true,decide_eq_true_eq,and_assoc] using ok

theorem scalar_observations (ok : scalarCheck s = Bool.true) :
    s.pythonInts = Bool.true ∧
    (s.args.numSms:Int)=s.sms ∧ (s.args.groupK:Int)=s.group ∧
    (s.args.m:Int)=s.m ∧ (s.args.n:Int)=s.n ∧
    (s.args.strideM:Int)=s.sm ∧ (s.args.strideN:Int)=s.sn ∧
    (s.args.bm:Int)=s.bm ∧ (s.args.bn:Int)=s.bn := by
  simp only [scalarCheck,Bool.and_eq_true,decide_eq_true_eq] at ok
  rcases ok with ⟨hi,hs,hg,hm,hn,hsm,hsn,hbm,hbn⟩
  simp [Scalars.args,hi,Int.toNat_of_nonneg,hs,hg,hm,hn,hsm,hsn,hbm,hbn]

def buffers (h : HostArgs) (p c l : Tensor) (ok : Metadata h p c l) : Wide.Buffers h := by
  rcases ok with ⟨hs,hsl,hg,hgl,hbm,hbml,hbn,hbnl,_,_,_,hm,hn,hsm,hsn,_,_,_,hp,hl,hc,wp,wc,wl,dpc,dpl,dcl,_,_⟩
  refine ⟨p.elements,c.elements,l.elements,p.base,c.base,l.base,hs,hsl,hg,hgl,hbm,hbn,hbml,hbnl,hm,hn,hsm,hsn,hp,hl,?_,wp.1,wc.1,wl.1,dpc,dpl,dcl⟩
  intro row hr col hc'
  have hh₁ := Nat.mul_le_mul_right h.strideM (show row≤h.m-1 by omega)
  have hh₂ := Nat.mul_le_mul_right h.strideN (show col≤h.n-1 by omega)
  rcases hc with hc|hc|hc <;> omega

def checkedBuffers (ok : check s p c l = Bool.true) : Wide.Buffers s.args :=
  buffers s.args p c l (checked_parts ok).2.2.2.2.1

theorem ready_words (ok : readyCheck h p l = Bool.true) (q : Nat)
    (hq : q<l.elements) (ready : l.word q=1) :
    (q+1)*(h.bm*h.bn)≤p.elements ∧ ∀ lane<h.bm*h.bn, p.word (q*(h.bm*h.bn)+lane)=0 := by
  have hh := List.all_eq_true.mp ok q (List.mem_range.mpr hq)
  simpa only [ready,BEq.rfl,↓reduceIte,Bool.and_eq_true,decide_eq_true_eq,
    List.all_eq_true,List.mem_range,beq_iff_eq] using hh

theorem checked_ready (b : Backend V) (decode : BitVec 32 → V) (zero : decode 0=b.zero)
    (ok : check s p c l = Bool.true) : Wide.PublishedReady b s.args (memory decode p c l) := by
  intro q hq lane hl
  have hmem : (if q<l.elements then l.word q else 0)=1 := by
    simpa only [memory,Int.natCast_nonneg,true_and,Int.toNat_natCast] using hq
  have hinside : q<l.elements := by
    by_contra hn
    simp [hn] at hmem
  have hw : l.word q=1 := by simpa only [if_pos hinside] using hmem
  have hz := (ready_words (checked_parts ok).2.2.2.2.2 q hinside hw).2 lane hl
  have ha : Wide.pAddress s.args q lane = ((q*(s.args.bm*s.args.bn)+lane : Nat) : Int) := by
    simp [Wide.pAddress,Nat.cast_add,Nat.cast_mul,Nat.mul_assoc]
  change decode (p.word (Wide.pAddress s.args q lane).toNat)=b.zero
  rw [ha,Int.toNat_natCast,hz,zero]

/-- Full pointer arithmetic, including output pointers in masked-off lanes,
is bounded and aligned. Accessed words additionally satisfy Buffers capacities. -/
theorem formed_output_bounds (ok : check s p c l = Bool.true)
    (pid lane : Nat) (hp : pid<s.args.numSms) (hl : lane<s.args.bm*s.args.bn) :
    let off := (lane/s.args.bn)*s.args.strideM+(pid*s.args.bn+lane%s.args.bn)*s.args.strideN
    off<2^63 ∧ c.base+4*off+4≤2^64 ∧ (c.base+4*off)%4=0 := by
  have facts := (checked_parts ok).2.2.2.2.1
  have buf := checkedBuffers ok
  have row : lane/s.args.bn<s.args.bm := (Nat.div_lt_iff_lt_mul buf.tileNPositive).mpr hl
  have rem : lane%s.args.bn<s.args.bn := Nat.mod_lt _ buf.tileNPositive
  have col : pid*s.args.bn+lane%s.args.bn<s.args.numSms*s.args.bn := by
    have hh := Nat.mul_le_mul_right s.args.bn (show pid+1≤s.args.numSms by omega)
    simp only [Nat.add_mul,Nat.one_mul] at hh
    omega
  have rm := Nat.mul_le_mul_right s.args.strideM (show lane/s.args.bn≤s.args.bm-1 by omega)
  have cn := Nat.mul_le_mul_right s.args.strideN (show pid*s.args.bn+lane%s.args.bn≤s.args.numSms*s.args.bn-1 by omega)
  rcases facts with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,wc,_,_,_,_,hb,hb'⟩
  dsimp only
  refine ⟨by omega,by omega,?_⟩
  have ha := wc.2.1
  simp [Nat.add_mod,Nat.mul_mod,ha]

theorem formed_output_exact (ok : check s p c l = Bool.true)
    (pid lane : Nat) (hp : pid<s.args.numSms) (hl : lane<s.args.bm*s.args.bn) :
    Wide.outputOffset64 s.args pid lane =
      ((lane/s.args.bn)*s.args.strideM+(pid*s.args.bn+lane%s.args.bn)*s.args.strideN : Nat) := by
  obtain ⟨hr,hc⟩ := Wide.coordinates64 s.args (checkedBuffers ok) pid lane hp
  have hb := (formed_output_bounds ok pid lane hp hl).1
  unfold Wide.outputOffset64
  rw [hr,hc]
  simp only [← BitVec.ofNat_mul]
  rw [Wide.signed64_nat _ (by omega),Wide.signed64_nat _ (by omega)]
  push_cast
  rfl

theorem window_access (w : Window t) (index byte : Nat) (hi : index<t.elements) (hb : byte<4) :
    t.base≤t.base+4*index+byte ∧ t.base+4*index+byte<t.base+4*t.elements ∧
    t.base+4*index+byte<2^64 ∧ (t.base+4*index)%4=0 := by
  refine ⟨by omega,by omega,by have := w.2.2; omega,?_⟩
  simp [Nat.add_mod,w.2.1]

#axiomsClean checked_parts
#axiomsClean scalar_observations
#axiomsClean checked_ready
#axiomsClean formed_output_bounds
#axiomsClean formed_output_exact
#axiomsClean window_access
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Admission
