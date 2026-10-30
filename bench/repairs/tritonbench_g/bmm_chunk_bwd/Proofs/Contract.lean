import bench.repairs.tritonbench_g.bmm_chunk_bwd.Contract
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.BmmChunkBwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section
set_option maxHeartbeats 3000000

theorem checkInputs_iff (p : Metadata) : checkInputs p=.ok () ↔ InputContract p := by
  have fields : InputContract p ↔
    (∀ r,Used p r → (p.tensor r).layout=.strided) ∧
    (∀ r,Used p r → (p.tensor r).dtype=.floating .fp32) ∧
    (∀ r,Used p r → (p.tensor r).device.kind="cuda") ∧
    (∀ r,Used p r → (p.tensor r).device=(p.tensor .a).device) ∧ Ranks p ∧ 0<chunk p ∧
    Shapes p ∧ ResidualShape p ∧ OutputShape p ∧
    (∀ r,Used p r → (p.tensor r).isContiguous=Bool.true) ∧ Strides p ∧
    (∀ r,Used p r → (p.tensor r).isNeg=Bool.false) ∧ Extents p ∧ Grid p ∧
    (∀ r,Used p r → 0<(p.tensor r).numel → base p r%4=0) ∧ Separate p := by
    constructor
    · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.device,h.ranks,h.chunk,h.shapes,h.residual,
        h.output,h.contiguous,h.strides,h.noNeg,h.extents,h.grid,h.alignment,h.separate⟩
    · rintro ⟨a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,q⟩; exact ⟨a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,q⟩
  rw [fields]
  simp [checkInputs,ite_eq_iff]

theorem checkArguments_iff (p : Option Metadata) : checkArguments p=.ok () ↔ Admitted p := by
  cases p <;> simp [checkArguments,Admitted,checkInputs_iff]

theorem group_count (p : Metadata) : (call p).groupCount=groups p := by
  unfold TypedSource.HostCall.groupCount call groups
  split <;> rfl

theorem input_shape (p : Metadata) (h : InputContract p) : (call p).tensorShape=(p.tensor .a).shape := by
  rcases h.ranks.1 with h3|h4
  · rcases hs : (p.tensor .a).shape with _|⟨a,xs⟩ <;> simp_all
    rcases xs with _|⟨b,xs⟩ <;> simp_all
    rcases xs with _|⟨c,xs⟩ <;> simp_all
    have nil : xs=[] := by simpa using h3
    subst xs
    simp [call,TypedSource.HostCall.tensorShape,grouped,batch,seqlen,groups,width,hs]
  · rcases hs : (p.tensor .a).shape with _|⟨a,xs⟩ <;> simp_all
    rcases xs with _|⟨b,xs⟩ <;> simp_all
    rcases xs with _|⟨c,xs⟩ <;> simp_all
    rcases xs with _|⟨d,xs⟩ <;> simp_all
    have nil : xs=[] := by simpa using h4
    subst xs
    simp [call,TypedSource.HostCall.tensorShape,grouped,batch,seqlen,groups,width,hs]

theorem input_size (p : Metadata) (h : InputContract p) :
    (p.tensor .a).numel=batch p*seqlen p*groups p*width p := by
  unfold VeriTile.Torch.TensorMetadata.numel
  rw [←input_shape p h]
  cases hg : grouped p <;> simp [call,TypedSource.HostCall.tensorShape,groups,hg,Nat.mul_assoc]

theorem dout_size (p : Metadata) (h : InputContract p) :
    (p.tensor .dout).numel=batch p*chunks p*groups p*chunk p*chunk p := by
  unfold VeriTile.Torch.TensorMetadata.numel
  rw [h.shapes.1]
  cases hg : grouped p <;> simp [call,TypedSource.HostCall.doutShape,groups,hg,Nat.mul_assoc]

theorem tensorStrideEq_iff (a b : TypedSource.TensorStrides) : TensorStrideEq a b ↔ a=b := by
  cases a; cases b; simp [TensorStrideEq,TypedSource.TensorStrides.mk.injEq]
theorem doutStrideEq_iff (a b : TypedSource.DoutStrides) : DoutStrideEq a b ↔ a=b := by
  cases a; cases b; simp [DoutStrideEq,TypedSource.DoutStrides.mk.injEq]

#axiomsClean checkArguments_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd
