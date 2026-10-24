import bench.repairs.tritonbench_g.quantize_kv_transform.Contract
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.Host
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.QuantizeKvTransform
noncomputable section
set_option maxHeartbeats 4000000

theorem checkInputs_iff (m : Inputs) (w : Nat → MetadataWord) :
    checkInputs m w=.ok () ↔ InputContract m w := by
  have fields : InputContract m w ↔
    (∀ r, (m r).layout=.strided) ∧ Types m ∧ (∀ r, (m r).device.kind="cuda") ∧
    (∀ r, (m r).device=(m .input).device) ∧ Ranks m ∧
    (∀ r, (m r).isContiguous=Bool.true) ∧ (∀ r, (m r).isNeg=Bool.false) ∧
    (0<(request m w).heads ∧ 0<(request m w).columns) ∧ Shapes m (request m w) ∧
    (request m w).tokens<2^31 ∧ (∀ r, (m r).numel<2^63) ∧
    headBlock (request m w)*columnBlock (request m w)≤2^20 ∧
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
  refine ⟨h.width.1,h.width.2,h.shapes.1,?_,?_,h.unique⟩
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

theorem sizes (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) :
    ∀ r, elements (request m w) r=elementBytes r*(m r).numel := by
  have si := triple (m .input).shape h.rank.1
  have so := triple (m .output).shape h.rank.2.2.1
  have tails := h.shapes.2.1
  rw [si,so] at tails
  simp only [List.drop_succ_cons,List.drop_zero,List.cons.injEq,and_true] at tails
  intro r
  cases r with
  | input =>
    simp only [elements,elementBytes,request,VeriTile.Torch.TensorMetadata.numel]
    rw [si]
    simp [Nat.mul_assoc]
  | destination =>
    simp only [elements,elementBytes,request,Nat.mul_one]
    split_ifs <;> omega
  | output =>
    simp only [elements,elementBytes,request,VeriTile.Torch.TensorMetadata.numel,Nat.one_mul]
    rw [so]
    simp only [List.prod_cons,List.prod_nil,Nat.mul_one]
    rw [tails.1,tails.2]
    simp [Nat.mul_assoc]
  | scale =>
    simp only [elements,elementBytes,VeriTile.Torch.TensorMetadata.numel]
    rw [h.shapes.2.2]
    simp

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

theorem direct_metadata (m : Inputs) (w : Nat → MetadataWord) (objects : Role → Nat)
    (heap : Heap) (binding : TensorBinding m w heap) : Metadata (request m w) (directLayout m objects) heap := by
  intro s hs
  simpa [request,destAddress,directLayout,Tensor.data,Nat.mul_comm] using binding.metadata s hs

#axiomsClean checkArguments_iff
#axiomsClean checked_domain
#axiomsClean direct_allocations
#axiomsClean direct_metadata
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvTransform
