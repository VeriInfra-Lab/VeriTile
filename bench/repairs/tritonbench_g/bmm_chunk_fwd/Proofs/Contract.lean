import bench.repairs.tritonbench_g.bmm_chunk_fwd.Contract
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.BmmChunkFwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
set_option maxHeartbeats 3000000

theorem checkInputs_iff (p : Metadata) : checkInputs p=.ok () ↔ InputContract p := by
  have fields : InputContract p ↔
    (∀ r,Used p r → (p.tensor r).layout=.strided) ∧ Types p ∧ OutputType p ∧
    (∀ r,Used p r → (p.tensor r).device.kind="cuda") ∧
    (∀ r,Used p r → (p.tensor r).device=(p.tensor .a).device) ∧ Shapes p ∧ SeqShape p ∧
    (∀ r,Used p r → (p.tensor r).isContiguous=Bool.true) ∧ Strides p ∧
    (∀ r,Used p r → (p.tensor r).isNeg=Bool.false) ∧ (p.chunkIsInt=Bool.true ∧ 0<p.chunk) ∧
    p.causalIsBool=Bool.true ∧ Extents p ∧ Grid p ∧
    (∀ r,Used p r → 0<(p.tensor r).numel → base p r%bytes r=0) ∧ Separate p := by
    constructor
    · intro h
      exact ⟨h.layout,h.dtype,h.output,h.cuda,h.device,h.shapes,h.sequence,h.contiguous,h.strides,h.noNeg,
        h.chunk,h.causal,h.extents,h.grid,h.alignment,h.separate⟩
    · rintro ⟨a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,q⟩; exact ⟨a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,q⟩
  rw [fields]
  simp [checkInputs,ite_eq_iff]

theorem checkArguments_iff (p : Option Metadata) : checkArguments p=.ok () ↔ Admitted p := by
  cases p <;> simp [checkArguments,Admitted,checkInputs_iff]

theorem contiguous_prepared (h : Host.Shape) : Host.prepared h (Host.contiguous h)=Host.contiguous h := by
  simp only [Host.prepared,Host.copied,Host.contiguous,ne_eq,not_true_eq_false,false_and,decide_false,Bool.false_eq_true,↓reduceIte]
  cases h.grouped <;> rfl

theorem input_shape (p : Metadata) (h : InputContract p) : (shape p).inputShape=(p.tensor .a).shape := by
  rcases h.shapes.1 with h3|h4
  · rcases hs : (p.tensor .a).shape with _|⟨a,xs⟩ <;> simp_all
    rcases xs with _|⟨b,xs⟩ <;> simp_all
    rcases xs with _|⟨c,xs⟩ <;> simp_all
    have nil : xs=[] := by simpa using h3
    subst xs
    simp [shape,Host.Shape.inputShape,hs]
  · rcases hs : (p.tensor .a).shape with _|⟨a,xs⟩ <;> simp_all
    rcases xs with _|⟨b,xs⟩ <;> simp_all
    rcases xs with _|⟨c,xs⟩ <;> simp_all
    rcases xs with _|⟨d,xs⟩ <;> simp_all
    have nil : xs=[] := by simpa using h4
    subst xs
    simp [shape,Host.Shape.inputShape,hs]

theorem input_size (p : Metadata) (h : InputContract p) :
    (p.tensor .a).numel=(shape p).batch*(shape p).seqlen*(shape p).groups*(shape p).K := by
  unfold VeriTile.Torch.TensorMetadata.numel
  rw [←input_shape p h]
  cases hg : (shape p).grouped <;> simp [Host.Shape.inputShape,Host.Shape.groups,hg,Nat.mul_assoc]

theorem seq_shape (p : Metadata) (h : InputContract p) (hs : p.hasSeq=Bool.true) :
    (p.tensor .seq).shape=[(shape p).batch,(shape p).seqlen] := by
  rw [h.sequence hs,←input_shape p h]
  cases hg : (shape p).grouped <;> simp [Host.Shape.inputShape,hg]

#axiomsClean checkArguments_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd
