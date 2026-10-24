import bench.repairs.tritonbench_g.chunk_bwd_dqkg.HostContract
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Host
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.DirectReal

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg
open VeriTile.Triton Whole
noncomputable section

/-- Whole host result for the repaired FP32 source with numerical precision
erased. The specification contains only pre-call logical inputs and finite
gradient formulas. It includes every returned element and the complete frame.
This theorem is deliberately separate from native floating-point conformance. -/
theorem correct (c : Config) (device : Nat) (before : BlockState) (alloc : Allocation before)
    (order : List (CTA c)) (ho : order.Perm (allCTAs c)) :
    ∃ result,invoke c device before alloc order=some result ∧
      result.tensors=outputDescriptors c device ∧
      (∀ (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K),result.state.readMem "dq" ((h.val*c.T+t.val)*c.K+e.val)=
        Spec.dq c (Spec.views c before.readMem) h.val (t.val/64) (t.val%64) e.val) ∧
      (∀ (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K),result.state.readMem "dk" ((h.val*c.T+t.val)*c.K+e.val)=
        Spec.dk c (Spec.views c before.readMem) h.val (t.val/64) (t.val%64) e.val) ∧
      (∀ (k : Fin c.NK) (h : Fin c.heads) (t : Fin c.T),result.state.readMem "dg" ((k.val*c.heads+h.val)*c.T+t.val)=
        Spec.dg c (Spec.views c before.readMem) k.val h.val (t.val/64) (t.val%64)) ∧
      (∀ (region : RegionName) (off : Nat),region≠"dq" → region≠"dk" → region≠"dg" →
        result.state.mem region off=before.mem region off) ∧
      BlockState.WriteWithin (gridWrites c (allCTAs c)) alloc.state result.state := by
  obtain ⟨final,he,hq,hk,hg,hf⟩ := whole_result c alloc.state order ho
  have hi := allocation_inputs before alloc
  refine ⟨⟨final,outputDescriptors c device⟩,by simp [invoke,he],rfl,?_,?_,?_,?_,hf⟩
  · intro h t e
    calc
      _ = expectedQ c before h t e := by simpa only [expectedQ,hi] using hq h t e
      _ = _ := Spec.logical_q c before h t e
  · intro h t e
    calc
      _ = expectedK c before h t e := by simpa only [expectedK,hi] using hk h t e
      _ = _ := Spec.logical_k c before h t e
  · intro k h t
    calc
      _ = expectedG c before k h t := by simpa only [expectedG,hi] using hg k h t
      _ = _ := Spec.logical_g c before k h t
  · intro region off hq hk hg
    have hn : ¬gridWrites c (allCTAs c) (region,off) := by
      rintro ⟨a,_,ha⟩
      rw [writes_cases] at ha
      rcases ha with ⟨hr,_⟩|⟨hr,_⟩|⟨hr,_⟩
      · exact hq hr
      · exact hk hr
      · exact hg hr
    exact (hf region off hn).symm.trans (alloc.frame region off hq hk hg)

#axiomsClean correct
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Direct
open VeriTile.Triton Whole
noncomputable section

structure RealResult where
  state : BlockState
  tensors : ReturnedTensor × ReturnedTensor × ReturnedTensor

def realWrapper (p : Metadata) (a : Allocation p) (trials : List (Fin 2)) (chosen : Fin 2)
    (data : RealData) : Except InputError (Option RealResult) := do
  checkInputs p
  checkOutputs p a
  pure ((executeReal p a trials chosen data).map fun state =>
    ⟨state,(returned p a .dq,returned p a .dk,returned p a .dg)⟩)

/-- Direct host checks, actual allocation bases, complete flattened source,
arbitrary-Real gradient target, and exact memory frame across every trial. -/
theorem chunk_bwd_dqkg_real (p : Metadata) (h : checkInputs p=.ok ()) (a : Allocation p)
    (outputs : checkOutputs p a=.ok ()) (trials : List (Fin 2)) (chosen : Fin 2) (data : RealData) :
    ∃ result,realWrapper p a trials chosen data=.ok (some result) ∧
      result.tensors=(returned p a .dq,returned p a .dk,returned p a .dg) ∧
      (∀ (hd : Fin (config p).heads) (t : Fin (config p).T) (e : Fin (config p).K),
        result.state.readMem (flat p a).flat (result.tensors.1.pointer/4+(hd.val*(config p).T+t.val)*(config p).K+e.val)=
          Spec.dq (config p) (Spec.views (config p) data) hd.val (t.val/64) (t.val%64) e.val) ∧
      (∀ (hd : Fin (config p).heads) (t : Fin (config p).T) (e : Fin (config p).K),
        result.state.readMem (flat p a).flat (result.tensors.2.1.pointer/4+(hd.val*(config p).T+t.val)*(config p).K+e.val)=
          Spec.dk (config p) (Spec.views (config p) data) hd.val (t.val/64) (t.val%64) e.val) ∧
      (∀ (k : Fin (config p).NK) (hd : Fin (config p).heads) (t : Fin (config p).T),
        result.state.readMem (flat p a).flat (result.tensors.2.2.pointer/4+(k.val*(config p).heads+hd.val)*(config p).T+t.val)=
          Spec.dg (config p) (Spec.views (config p) data) k.val hd.val (t.val/64) (t.val%64)) ∧
      BlockState.WriteWithin (flatWrites (config p) (flat p a))
        ((flat p a).flattenState (initialized p data)) result.state ∧
      (Nonempty p → sourceLayout p=(config p).layout ∧ Guard (config p) ∧
        ∀ r off,off<envelope (config p) → pointer p a r+4*off<2^64) := by
  have admitted := (checkInputs_iff p).mp h
  obtain ⟨final,run,hq,hk,hg,frame⟩ := real_complete p admitted a trials chosen data
  refine ⟨⟨final,(returned p a .dq,returned p a .dk,returned p a .dg)⟩,?_,rfl,?_,?_,?_,frame,?_⟩
  · simp [realWrapper,h,outputs,run] <;> rfl
  · simpa [returned,flat,FlatAlloc.addr,ofRegion,Nat.add_assoc] using hq
  · simpa [returned,flat,FlatAlloc.addr,ofRegion,Nat.add_assoc] using hk
  · simpa [returned,flat,FlatAlloc.addr,ofRegion,Nat.add_assoc] using hg
  · intro nonempty
    exact ⟨sourceLayout_eq p admitted nonempty,native_guard p admitted nonempty,
      formed_safe p admitted a nonempty ((checkOutputs_iff p a).mp outputs)⟩

theorem returned_identity (p : Metadata) (a : Allocation p) :
    (returned p a .dq).object=a.objects .dq ∧ (returned p a .dk).object=a.objects .dk ∧
    (returned p a .dg).object=a.objects .dg := ⟨rfl,rfl,rfl⟩

#axiomsClean chunk_bwd_dqkg_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Direct
