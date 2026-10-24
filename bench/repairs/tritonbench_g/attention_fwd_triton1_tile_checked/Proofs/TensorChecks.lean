import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.Tensor

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

structure MainFacts (c : Shape) (p : Payload) : Prop where
  shape : p.tensor.metadata.shape=Host.mainShape c
  contiguous : p.tensor.metadata.isContiguous=Bool.true
  ordinary : p.tensor.metadata.isNeg=Bool.false

theorem input_facts (c : Shape) (abi : ABI c) (t : Tensors) (b : TensorBinding c abi t)
    (storeFlag ifcond : Bool) (h : InputContract c (t.arguments storeFlag ifcond) b.pointers)
    (p : Payload) (hp : p∈[t.q,t.k,t.v]) : MainFacts c p := by
  have hm : AttentionFwdTriton1.Checked.metadataChecks c (t.arguments storeFlag ifcond)=Bool.true := by
    have hi := h.1
    simpa [AttentionFwdTriton1.Checked.inputChecks,Tensors.arguments] using hi
  have checked := (AttentionFwdTriton1.Checked.metadataChecks_iff _ _).1 hm p.observed
  have mem : p.observed∈[(t.arguments storeFlag ifcond).q,
      (t.arguments storeFlag ifcond).k,(t.arguments storeFlag ifcond).v] := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hp
    rcases hp with rfl|rfl|rfl <;> simp [Tensors.arguments]
  exact ⟨(checked mem).1,(checked mem).2.2.2.2.1,(checked mem).2.2.2.2.2.2⟩

theorem main_numel (c : Shape) (p : Payload) (shape : p.tensor.metadata.shape=Host.mainShape c) :
    p.tensor.metadata.numel=c.mainSize := by
  simp [VeriTile.Torch.TensorMetadata.numel,shape,Host.mainShape,Shape.mainSize,
    Shape.heads,Shape.headSize,Nat.mul_assoc]

theorem history_numel (c : Shape) (p : Payload) (shape : p.tensor.metadata.shape=Host.historyShape c) :
    p.tensor.metadata.numel=c.historySize := by
  simp [VeriTile.Torch.TensorMetadata.numel,shape,Host.historyShape,Shape.historySize,
    Shape.heads,Shape.historyHead,Nat.mul_assoc]

structure PointerBases (c : Shape) (abi : ABI c) (t : Tensors) : Prop where
  q : t.q.tensor.dataPtr.toNat=4*abi.q.base
  k : t.k.tensor.dataPtr.toNat=4*abi.k.base
  v : t.v.tensor.dataPtr.toNat=4*abi.v.base
  h : t.history.tensor.dataPtr.toNat=4*abi.h.base
  o : t.output.tensor.dataPtr.toNat=4*abi.o.base

theorem pointer_base (base word extent : Nat) (bind : word=base/4)
    (check : AttentionFwdTriton1.Checked.pointerCheck base extent=Bool.true) : base=4*word := by
  simp only [AttentionFwdTriton1.Checked.pointerCheck,decide_eq_true_eq] at check
  have h := Nat.mod_add_div base 4
  omega

theorem pointer_bases (c : Shape) (abi : ABI c) (t : Tensors) (b : TensorBinding c abi t)
    (storeFlag ifcond : Bool) (h : InputContract c (t.arguments storeFlag ifcond) b.pointers)
    (live : ¬Host.Empty c) : PointerBases c abi t := by
  have hs := h.2.1
  simp only [AttentionFwdTriton1.Checked.setupChecks,show ¬(c.T=0 ∨ c.heads=0) from live,
    ↓reduceIte,Bool.and_eq_true,AttentionFwdTriton1.Checked.pointerChecks] at hs
  rcases hs.2 with ⟨⟨⟨⟨hq,hk⟩,hv⟩,hh⟩,ho⟩
  exact ⟨pointer_base _ _ _ b.qBase hq,pointer_base _ _ _ b.kBase hk,
    pointer_base _ _ _ b.vBase hv,pointer_base _ _ _ b.hBase hh,pointer_base _ _ _ b.oBase ho⟩

theorem physical_logical {A} [Inhabited A] (before : Cells A) (p : Payload) (base i : Nat)
    (bind : p.tensor.dataPtr.toNat=4*base) (contiguous : p.tensor.metadata.isContiguous=Bool.true)
    (hi : i<p.tensor.metadata.numel) :
    (before (4*(base+i))).read .fp32=logical before p i := by
  unfold logical
  rw [←payload_address p contiguous i hi,bind,Nat.mul_add]

theorem inputs_logical {A} [Inhabited A] (c : Shape) (abi : ABI c) (t : Tensors)
    (b : TensorBinding c abi t) (storeFlag ifcond : Bool)
    (h : InputContract c (t.arguments storeFlag ifcond) b.pointers) (before : Cells A)
    (head row j : Nat) (hh : head<c.heads) (hr : row<c.T) (hj : j<c.D) :
    (tensorInputs c abi before).q head row j=logical before t.q (c.mainOffset head row j) ∧
    (tensorInputs c abi before).k head row j=logical before t.k (c.mainOffset head row j) ∧
    (tensorInputs c abi before).v head row j=logical before t.v (c.mainOffset head row j) := by
  have live : ¬Host.Empty c := by unfold Host.Empty; omega
  have ptr := pointer_bases c abi t b storeFlag ifcond h live
  have q := input_facts c abi t b storeFlag ifcond h t.q (by simp)
  have k := input_facts c abi t b storeFlag ifcond h t.k (by simp)
  have v := input_facts c abi t b storeFlag ifcond h t.v (by simp)
  have bound := c.main_bound head row j hh hr hj
  exact ⟨physical_logical before t.q _ _ ptr.q q.contiguous (by rwa [main_numel c t.q q.shape]),
    physical_logical before t.k _ _ ptr.k k.contiguous (by rwa [main_numel c t.k k.shape]),
    physical_logical before t.v _ _ ptr.v v.contiguous (by rwa [main_numel c t.v v.shape])⟩

#axiomsClean pointer_bases
#axiomsClean inputs_logical
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
