import bench.repairs.tritonbench_g.quantize_kv_copy.Contract
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.Host
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.QuantizeKvCopy
noncomputable section
set_option maxHeartbeats 4000000

theorem checkInputs_iff (m : Inputs) (w : Nat → MetadataWord) :
    checkInputs m w=.ok () ↔ InputContract m w := by
  have fields : InputContract m w ↔
    (∀ r, (m r).layout=.strided) ∧ Types m ∧ (∀ r, (m r).device.kind="cuda") ∧
    (∀ r, (m r).device=(m .input).device) ∧ Ranks m ∧
    (∀ r, (m r).isContiguous=Bool.true) ∧ (∀ r, (m r).isNeg=Bool.false) ∧
    Width m ∧ Shapes m (request m w) ∧
    ((request m w).tokens<2^31 ∧ (request m w).headCount<2^16) ∧ (∀ r, (m r).numel<2^63) ∧
    groupBlock (request m w)*8≤2^20 ∧
    (∀ r, 0<(m r).numel → base m r%elementBytes r=0) ∧ Separate m ∧ Bounds (request m w) w ∧
    Unique (request m w) := by
    constructor
    · intro h
      exact ⟨h.layout,h.dtype,h.cuda,h.device,h.rank,h.contiguous,h.noNeg,h.width,h.shapes,
        h.grid,h.extent,h.tile,h.alignment,h.separate,h.bounds,h.unique⟩
    · rintro ⟨a,b,c,d,e,f,g,h,i,j,k,l,n,o,p,q⟩; exact ⟨a,b,c,d,e,f,g,h,i,j,k,l,n,o,p,q⟩
  rw [fields]
  simp [checkInputs,ite_eq_iff]

theorem checkArguments_iff (m : Option Inputs) (w : Nat → MetadataWord) :
    checkArguments m w=.ok () ↔ Admitted m w := by
  cases m <;> simp [checkArguments,Admitted,checkInputs_iff]

theorem checked_domain (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) : Domain (request m w) := by
  have hg : 0<(request m w).groups := by
    have width := h.width
    simp only [Width] at width
    change 0<(m .input).shape.getD 2 0 / 8
    omega
  refine ⟨h.width.1,hg,h.shapes.1,?_,?_,h.unique⟩
  · intro s hs
    have bound := h.bounds s hs
    simp only [request] at bound ⊢
    omega
  · intro s hs
    have nonneg := (h.bounds s hs).1
    have upper := (w s).toInt_lt
    simp only [request]
    omega

theorem triple (xs : List Nat) (h : xs.length=3) :
    xs=[xs.getD 0 0,xs.getD 1 0,xs.getD 2 0] := by
  rcases xs with _ | ⟨a,xs⟩ <;> simp at h ⊢
  rcases xs with _ | ⟨b,xs⟩ <;> simp at h ⊢
  rcases xs with _ | ⟨c,xs⟩ <;> simp at h ⊢
  simpa using h

theorem dimensions (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) :
    (m .input).shape=[(request m w).sourceRows,(request m w).headCount,(request m w).groups*8] ∧
    (m .output).shape=[(request m w).capacity,(request m w).headCount,(request m w).groups*8] := by
  have si := triple (m .input).shape h.rank.1
  have so := triple (m .output).shape h.rank.2.2.1
  have tails := h.shapes.2.1
  rw [si,so] at tails
  simp only [List.drop_succ_cons,List.drop_zero,List.cons.injEq,and_true] at tails
  have hd : (m .input).shape.getD 2 0 / 8 * 8=(m .input).shape.getD 2 0 :=
    Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h.width.2.2)
  constructor
  · change (m .input).shape=[_,_,(m .input).shape.getD 2 0 / 8*8]
    rw [hd]
    exact si
  · change (m .output).shape=[_,_,(m .input).shape.getD 2 0 / 8*8]
    simp only [request]
    rw [hd,←tails.1,←tails.2]
    exact so

theorem sizes (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) :
    ∀ r, elements (request m w) r=elementBytes r*(m r).numel := by
  intro r
  cases r with
  | input =>
    simp only [elements,elementBytes,Request.sourcePrograms,VeriTile.Torch.TensorMetadata.numel]
    rw [(dimensions m w h).1]
    simp [groupWidth,Nat.mul_assoc]
  | destination =>
    simp only [elements,elementBytes,request,Nat.mul_one]
    split_ifs <;> omega
  | output =>
    simp only [elements,elementBytes,Request.outputPrograms,VeriTile.Torch.TensorMetadata.numel]
    rw [(dimensions m w h).2]
    simp [groupWidth,Nat.mul_assoc]
  | scale =>
    simp only [elements,elementBytes,VeriTile.Torch.TensorMetadata.numel]
    rw [h.shapes.2.2]
    simp [Request.outputPrograms,Nat.mul_assoc]

theorem direct_allocations (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (objects : Role → Nat) : NativeAllocations (request m w) (directLayout m objects) := by
  constructor
  · intro r
    simp only [directLayout,Nat.zero_add,sizes m w h]
    exact le_rfl
  · intro r s distinct a ha hb
    have separation := h.separate r s distinct
    change base m r≤a ∧ a<base m r+elementBytes r*(m r).numel at ha
    change base m s≤a ∧ a<base m s+elementBytes s*(m s).numel at hb
    rcases separation with left|right <;> omega

theorem direct_raw_metadata (m : Inputs) (w : Nat → MetadataWord) (objects : Role → Nat)
    (heap : Heap) (binding : TensorBinding m w heap) : RawMetadata (request m w) (directLayout m objects) heap := by
  intro s hs
  simpa [request,directLayout,Tensor.data,Nat.mul_comm] using binding.metadata s hs

theorem direct_metadata (m : Inputs) (w : Nat → MetadataWord) (objects : Role → Nat)
    (heap : Heap) (binding : TensorBinding m w heap) : Metadata (request m w) (directLayout m objects) heap :=
  Host.metadata_from_int32 (request m w) (directLayout m objects) heap (direct_raw_metadata m w objects heap binding)

#axiomsClean checkArguments_iff
#axiomsClean checked_domain
#axiomsClean direct_allocations
#axiomsClean direct_metadata
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvCopy
