import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.QuantReal
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.QuantRaw
open VeriTile.Triton FP.Structural
noncomputable section
abbrev RealHeap := Nat → Nat → Cell ℝ

def realPublish (shape : Quantizer.Shape) (Q S Z : Nat) (heap : RealHeap) (v : Quantizer.Result ℝ) : RealHeap := fun r a =>
  if r=Q ∧ a<(shape.K/8)*shape.N then
    .mk .int (v.Q (a/shape.N) (a%shape.N)).toInt
  else if r=S ∧ a<shape.groups*shape.N then
    .mk .fp32 (v.S (a/shape.N) (a%shape.N))
  else if r=Z ∧ a<shape.groups*(shape.N/8) then
    .mk .int (v.Z (a/(shape.N/8)) (a%(shape.N/8))).toInt else heap r a


def Writes (shape : Quantizer.Shape) (Q S Z : Nat) (r a : Nat) : Prop :=
  (r=Q ∧ a<(shape.K/8)*shape.N) ∨
  (r=S ∧ a<shape.groups*shape.N) ∨
  (r=Z ∧ a<shape.groups*(shape.N/8))

def RealPost (shape : Quantizer.Shape) (x : Nat → Nat → ℝ) (Q S Z : Nat) (heap final : RealHeap) : Prop :=
  (∀ k n,k<shape.K/8 → n<shape.N →
    final Q (k*shape.N+n)=
      .mk .int (QuantReal.packedQ shape x k n).toInt) ∧
  (∀ g n,g<shape.groups → n<shape.N →
    final S (g*shape.N+n)=
      .mk .fp32 (QuantReal.scale shape x g n)) ∧
  (∀ g n,g<shape.groups → n<shape.N/8 →
    final Z (g*(shape.N/8)+n)=
      .mk .int (QuantReal.packedZ shape x g n).toInt) ∧
  (∀ r a,¬Writes shape Q S Z r a → final r a=heap r a)


theorem coordinate (rows cols r n : Nat) (hr : r<rows) (hn : n<cols) :
    r*cols+n<rows*cols ∧ (r*cols+n)/cols=r ∧ (r*cols+n)%cols=n := by
  refine ⟨MatmulKernel.tiled_bound r n rows cols hr hn,?_,?_⟩
  · rw [Nat.add_comm,Nat.add_mul_div_right _ _ (by omega),Nat.div_eq_of_lt hn,Nat.zero_add]
  · rw [Nat.add_comm,Nat.add_mul_mod_self_right,Nat.mod_eq_of_lt hn]


theorem real_complete (tree : Quantizer.Tree) (shape : Quantizer.Shape)
    (valid : shape.Valid) (covered : tree.Valid shape.group)
    (x : Nat → Nat → ℝ) (Q S Z : Nat) (qs : Q≠S) (qz : Q≠Z) (sz : S≠Z)
    (heap : RealHeap) :
    ∃ values final, values=Quantizer.source (QuantReal.ops tree) shape x ∧
      final=realPublish shape Q S Z heap values ∧ RealPost shape x Q S Z heap final := by
  let v := Quantizer.source (QuantReal.ops tree) shape x
  have target := QuantReal.whole_real tree shape valid covered x
  refine ⟨v,realPublish shape Q S Z heap v,rfl,rfl,?_,?_,?_,?_⟩
  · intro k n hk hn
    have c := coordinate (shape.K/8) shape.N k n hk hn
    simp only [realPublish,c.1,true_and,ite_true,c.2.1,c.2.2,v,target.1 k n hk hn]
  · intro g n hg hn
    have c := coordinate shape.groups shape.N g n hg hn
    simp only [realPublish,c.1,Ne.symm qs,false_and,ite_false,true_and,ite_true,c.2.1,c.2.2,v,target.2.1 g n hg hn]
  · intro g n hg hn
    have c := coordinate shape.groups (shape.N/8) g n hg hn
    simp only [realPublish,c.1,Ne.symm qz,Ne.symm sz,false_and,ite_false,
      true_and,ite_true,c.2.1,c.2.2,v,target.2.2.1 g n hg hn]
  · intro r a outside
    have q : ¬(r=Q ∧ a<(shape.K/8)*shape.N) := fun h => outside (Or.inl h)
    have s : ¬(r=S ∧ a<shape.groups*shape.N) := fun h => outside (Or.inr (Or.inl h))
    have z : ¬(r=Z ∧ a<shape.groups*(shape.N/8)) := fun h => outside (Or.inr (Or.inr h))
    simp only [realPublish,q,s,z,ite_false]

def nativeRun (shape : Quantizer.Shape) (view : View) (tree : Quantizer.Tree)
    (actual : QuantFP.Instructions) (initial : Physical.Bytes) : Physical.Bytes :=
  QuantHost.publish ⟨shape,view,0,tree⟩
    (QuantHost.scalarResult ⟨shape,view,0,tree⟩ actual initial) initial

def NativePost (shape : Quantizer.Shape) (view : View) (tree : Quantizer.Tree)
    (initial final : Physical.Bytes) : Prop :=
  (∀ k n,k<shape.K/8 → n<shape.N → ∀ byte : Fin 4,
    final "Qout" (4*(k*shape.N+n)+byte.val)=
      QuantHost.encode32 (Quantizer.targetPackedQ (QuantFP.ops QuantFP.software tree) shape (QuantHost.input ⟨shape,view,0,tree⟩ initial) k n) byte) ∧
  (∀ g n,g<shape.groups → n<shape.N → ∀ byte : Fin 4,
    final "Sout" (4*(g*shape.N+n)+byte.val)=
      QuantHost.encode32 (Quantizer.targetScale (QuantFP.ops QuantFP.software tree) shape (QuantHost.input ⟨shape,view,0,tree⟩ initial) g n).bits byte) ∧
  (∀ g n,g<shape.groups → n<shape.N/8 → ∀ byte : Fin 4,
    final "Zout" (4*(g*(shape.N/8)+n)+byte.val)=
      QuantHost.encode32 (Quantizer.targetPackedZ (QuantFP.ops QuantFP.software tree) shape (QuantHost.input ⟨shape,view,0,tree⟩ initial) g n) byte) ∧
  (∀ r b,¬QuantHost.Writes ⟨shape,view,0,tree⟩ r b → final r b=initial r b)

theorem native_complete (shape : Quantizer.Shape) (view : View) (tree : Quantizer.Tree)
    (valid : shape.Valid) (viewShape : view.rows=shape.K ∧ view.cols=shape.N)
    (viewValid : view.Valid) (covered : tree.Valid shape.group)
    (actual : QuantFP.Instructions) (ports : QuantFP.Obligations actual)
    (initial : Physical.Bytes) :
    ∃ final, final=nativeRun shape view tree actual initial ∧
      NativePost shape view tree initial final := by
  have result := QuantHost.whole_call ⟨shape,view,0,tree⟩
    ⟨valid,viewShape,viewValid,covered⟩ actual ports initial
  exact ⟨nativeRun shape view tree actual initial,rfl,
    result.1,result.2.1,result.2.2.1,result.2.2.2.1⟩

#axiomsClean real_complete
#axiomsClean native_complete
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.QuantRaw
