import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedGrid
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Physical
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

/-- Only allocation preservation and disjoint live tensor storage; no results. -/
structure Allocation (before : State α) where
  state : State α
  preserved : ∀ region off,region∉(["dq","dk","dg"] : List RegionName) → state.mem region off=before.mem region off

theorem allocation_inputs (before : State α) (a : Allocation before) : SameInputs a.state before := by
  intro region hr
  funext off
  apply a.preserved
  simp only [List.mem_cons,List.mem_nil_iff,or_false] at hr
  rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> decide

structure Result where
  state : State α
  tensors : PhysicalTensor × PhysicalTensor × PhysicalTensor

def invoke (M : Algebra α) (dot : Dot α) (c : Config) (device : Nat) (A : FlatAlloc)
    (before : State α) (alloc : Allocation before) (order : List (CTA c)) : Option (Result (α:=α)) :=
  (runGrid M dot c order alloc.state).map fun state=>⟨state,physicalDescriptors c device A⟩

/-- A physical read uses the same placement table as returned descriptors. -/
def physicalCell (A : FlatAlloc) (s : State α) (off : Nat) : Cell α :=
  match A.decode off with | some (region,i)=>s.mem region i | none=>.mk .real default

def materialize (A : FlatAlloc) (encode : Cell α → BitVec 32) (raw : Nat → BitVec 8)
    (s : State α) (i : Nat) : BitVec 8 :=
  match A.decode (i/4) with
  | some (region,off)=>byteOf (encode (s.mem region off)) (i%4)
  | none=>raw i

def BytesBound (A : FlatAlloc) (encode : Cell α → BitVec 32) (raw : Nat → BitVec 8)
    (s : State α) : Prop := ∀ i region off,A.decode (i/4)=some (region,off) →
      raw i=byteOf (encode (s.mem region off)) (i%4)

def ByteFootprint (c : Config) (A : FlatAlloc) (i : Nat) : Prop :=
  ∃ region off,A.decode (i/4)=some (region,off) ∧ gridWrites c (allCTAs c) (region,off)

theorem materialize_frame (c : Config) (A : FlatAlloc) (encode : Cell α → BitVec 32)
    (raw : Nat → BitVec 8) (s t : State α) (hb : BytesBound A encode raw s)
    (hf : ∀ region off,¬gridWrites c (allCTAs c) (region,off) → t.mem region off=s.mem region off)
    (i : Nat) (hi : ¬ByteFootprint c A i) : materialize A encode raw t i=raw i := by
  cases he : A.decode (i/4) with
  | none=>simp [materialize,he]
  | some pair=>
    rcases pair with ⟨region,off⟩
    simp only [materialize,he]
    rw [hf region off (fun hw=>hi ⟨region,off,he,hw⟩)]
    exact (hb i region off he).symm

/-- Whole typed call: returned bases, all logical values from pre-call inputs,
and the exact byte frame are bound to the one placement table. Physical
materialization is explicit; this theorem does not assert FP AST flattening. -/
theorem whole_call (M : Algebra α) (dot : Dot α) (c : Config) (device : Nat)
    (A : FlatAlloc) (ha : DenseAllocation c A) (before : State α) (alloc : Allocation before)
    (order : List (CTA c)) (ho : order.Perm (allCTAs c))
    (encode : Cell α → BitVec 32) (raw : Nat → BitVec 8)
    (hb : BytesBound A encode raw alloc.state) :
    ∃ result,invoke M dot c device A before alloc order=some result ∧
      result.tensors=physicalDescriptors c device A ∧
      (∀ (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K),
        physicalCell A result.state (result.tensors.1.byteBase/4+(h.val*c.T+t.val)*c.K+e.val)=
          .mk .fp32 (expectedQ M dot c before h t e)) ∧
      (∀ (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K),
        physicalCell A result.state (result.tensors.2.1.byteBase/4+(h.val*c.T+t.val)*c.K+e.val)=
          .mk .fp32 (expectedK M dot c before h t e)) ∧
      (∀ (k : Fin c.NK) (h : Fin c.heads) (t : Fin c.T),
        physicalCell A result.state (result.tensors.2.2.byteBase/4+(k.val*c.heads+h.val)*c.T+t.val)=
          .mk .real (expectedG M dot c before k h t)) ∧
      (∀ region off,¬gridWrites c (allCTAs c) (region,off) → result.state.mem region off=alloc.state.mem region off) ∧
      (∀ i,¬ByteFootprint c A i → materialize A encode raw result.state i=raw i) := by
  obtain ⟨final,he,hq,hk,hg,hf⟩:=whole_result M dot c alloc.state order ho
  have hin (a : CTA c):=results_inputs M dot c alloc.state before a (allocation_inputs before alloc)
  refine ⟨⟨final,physicalDescriptors c device A⟩,by simp [invoke,he],rfl,?_,?_,?_,hf,?_⟩
  · intro h t e
    have hd := A.decode_addr ha.apart (r:=("dq" : RegionName)) (o:=((h.val*c.T+t.val)*c.K+e.val))
      (by rw [ha.listed];simp [regions])
      (by rw [ha.sizes];simpa [extent] using pack_bound _ _ _ _ (pack_bound _ _ _ _ h.isLt t.isLt) e.isLt)
    simp only [FlatAlloc.addr,Nat.add_assoc] at hd
    have hv:=hq h t e
    simp only [expectedQ,(hin (qOwner c h t e)).1] at hv
    simpa [physicalDescriptors,outputDescriptors,placed,tensor,FlatAlloc.addr,Nat.add_assoc,physicalCell,hd,expectedQ,expectedK,expectedG] using hv
  · intro h t e
    have hd := A.decode_addr ha.apart (r:=("dk" : RegionName)) (o:=((h.val*c.T+t.val)*c.K+e.val))
      (by rw [ha.listed];simp [regions])
      (by rw [ha.sizes];simpa [extent] using pack_bound _ _ _ _ (pack_bound _ _ _ _ h.isLt t.isLt) e.isLt)
    simp only [FlatAlloc.addr,Nat.add_assoc] at hd
    have hv:=hk h t e
    simp only [expectedK,(hin (qOwner c h t e)).2.1] at hv
    simpa [physicalDescriptors,outputDescriptors,placed,tensor,FlatAlloc.addr,Nat.add_assoc,physicalCell,hd,expectedQ,expectedK,expectedG] using hv
  · intro k h t
    have hd := A.decode_addr ha.apart (r:=("dg" : RegionName)) (o:=((k.val*c.heads+h.val)*c.T+t.val))
      (by rw [ha.listed];simp [regions])
      (by rw [ha.sizes];simpa [extent] using pack_bound _ _ _ _ (pack_bound _ _ _ _ k.isLt h.isLt) t.isLt)
    simp only [FlatAlloc.addr,Nat.add_assoc] at hd
    have hv:=hg k h t
    simp only [expectedG,(hin (gOwner c k h t)).2.2] at hv
    simpa [physicalDescriptors,outputDescriptors,placed,tensor,FlatAlloc.addr,Nat.add_assoc,physicalCell,hd,expectedQ,expectedK,expectedG] using hv
  · exact fun i hi=>materialize_frame c A encode raw alloc.state final hb hf i hi

#axiomsClean whole_call

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
