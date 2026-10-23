import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.QuantTensor
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.QuantRealTensor
open VeriTile.Triton FP.Structural Checked
noncomputable section
abbrev Heap := Nat → Nat → Cell ℝ

def input (p : QuantMetadata) (t : QuantTensor.Tensors p) (heap : Heap) (k n : Nat) : ℝ :=
  (heap t.weight.storage.object (t.weight.physicalIndex (k*p.call.shape.N+n))).read .fp32

/-- Ideal-real interpretation of the eager library's three fresh contiguous
outputs. All values come from the complete source quantizer/packing loops. -/
def publish (p : QuantMetadata) (t : QuantTensor.Tensors p) (heap : Heap) (v : Quantizer.Result ℝ) : Heap := fun r a =>
  if r=t.q.observed.storageObject ∧ a<(p.call.shape.K/8)*p.call.shape.N then
    .mk .int (v.Q (a/p.call.shape.N) (a%p.call.shape.N)).toInt
  else if r=t.s.storage.object ∧ a<p.call.shape.groups*p.call.shape.N then
    .mk .fp32 (v.S (a/p.call.shape.N) (a%p.call.shape.N))
  else if r=t.z.observed.storageObject ∧ a<p.call.shape.groups*(p.call.shape.N/8) then
    .mk .int (v.Z (a/(p.call.shape.N/8)) (a%(p.call.shape.N/8))).toInt else heap r a

structure Result where
  memory : Heap
  returned : QuantTensor.PackedTensor × VeriTile.Torch.FP32.Tensor × QuantTensor.PackedTensor × Nat

def invoke (p : QuantMetadata) (t : QuantTensor.Tensors p) (raw : QuantTensor.Heap) (heap : Heap) : Except Error Result := do
  checkQuant p (QuantTensor.memory p t raw)
  let v := Quantizer.source (QuantReal.ops p.call.tree) p.call.shape (input p t heap)
  pure ⟨publish p t heap v,(t.q,t.s,t.z,p.call.shape.group)⟩

def Writes (p : QuantMetadata) (t : QuantTensor.Tensors p) (r a : Nat) : Prop :=
  (r=t.q.observed.storageObject ∧ a<(p.call.shape.K/8)*p.call.shape.N) ∨
  (r=t.s.storage.object ∧ a<p.call.shape.groups*p.call.shape.N) ∨
  (r=t.z.observed.storageObject ∧ a<p.call.shape.groups*(p.call.shape.N/8))

def Guarantee (p : QuantMetadata) (t : QuantTensor.Tensors p) (heap : Heap) (result : Result) : Prop :=
  result.returned=(t.q,t.s,t.z,p.call.shape.group) ∧
  (∀ k n,k<p.call.shape.K/8 → n<p.call.shape.N →
    result.memory t.q.observed.storageObject (k*p.call.shape.N+n)=
      .mk .int (QuantReal.packedQ p.call.shape (input p t heap) k n).toInt) ∧
  (∀ g n,g<p.call.shape.groups → n<p.call.shape.N →
    result.memory t.s.storage.object (g*p.call.shape.N+n)=
      .mk .fp32 (QuantReal.scale p.call.shape (input p t heap) g n)) ∧
  (∀ g n,g<p.call.shape.groups → n<p.call.shape.N/8 →
    result.memory t.z.observed.storageObject (g*(p.call.shape.N/8)+n)=
      .mk .int (QuantReal.packedZ p.call.shape (input p t heap) g n).toInt) ∧
  (∀ r a,¬Writes p t r a → result.memory r a=heap r a)

theorem coordinate (rows cols r n : Nat) (hr : r<rows) (hn : n<cols) :
    r*cols+n<rows*cols ∧ (r*cols+n)/cols=r ∧ (r*cols+n)%cols=n := by
  refine ⟨MatmulKernel.tiled_bound r n rows cols hr hn,?_,?_⟩
  · rw [Nat.add_comm,Nat.add_mul_div_right _ _ (by omega),Nat.div_eq_of_lt hn,Nat.zero_add]
  · rw [Nat.add_comm,Nat.add_mul_mod_self_right,Nat.mod_eq_of_lt hn]

theorem whole_tensor (p : QuantMetadata) (t : QuantTensor.Tensors p) (raw : QuantTensor.Heap) (heap : Heap)
    (h : QuantInputContract p (QuantTensor.memory p t raw)) (tree : ReductionBinding p.call) :
    ∃ result,invoke p t raw heap=.ok result ∧ Guarantee p t heap result := by
  have valid := (quant_admitted p (QuantTensor.memory p t raw) h tree).valid
  let v := Quantizer.source (QuantReal.ops p.call.tree) p.call.shape (input p t heap)
  have target := QuantReal.whole_real p.call.tree p.call.shape valid.shape valid.tree (input p t heap)
  refine ⟨⟨publish p t heap v,(t.q,t.s,t.z,p.call.shape.group)⟩,?_,rfl,?_,?_,?_,?_⟩
  · simp [invoke,(checkQuant_iff _ _).mpr h,v]
  · intro k n hk hn
    have c := coordinate (p.call.shape.K/8) p.call.shape.N k n hk hn
    simp only [publish,c.1,true_and,ite_true,c.2.1,c.2.2,v,target.1 k n hk hn]
  · intro g n hg hn
    have c := coordinate p.call.shape.groups p.call.shape.N g n hg hn
    simp only [publish,c.1,Ne.symm t.distinctQS,false_and,ite_false,true_and,ite_true,c.2.1,c.2.2,v,target.2.1 g n hg hn]
  · intro g n hg hn
    have c := coordinate p.call.shape.groups (p.call.shape.N/8) g n hg hn
    simp only [publish,c.1,Ne.symm t.distinctQZ,Ne.symm t.distinctSZ,false_and,ite_false,
      true_and,ite_true,c.2.1,c.2.2,v,target.2.2.1 g n hg hn]
  · intro r a outside
    have q : ¬(r=t.q.observed.storageObject ∧ a<(p.call.shape.K/8)*p.call.shape.N) := fun h => outside (Or.inl h)
    have s : ¬(r=t.s.storage.object ∧ a<p.call.shape.groups*p.call.shape.N) := fun h => outside (Or.inr (Or.inl h))
    have z : ¬(r=t.z.observed.storageObject ∧ a<p.call.shape.groups*(p.call.shape.N/8)) := fun h => outside (Or.inr (Or.inr h))
    simp only [publish,q,s,z,ite_false]

#specNonCircular Guarantee avoiding [invoke,publish,Quantizer.source]
#axiomsClean whole_tensor
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.QuantRealTensor
