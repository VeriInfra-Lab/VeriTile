import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.TensorEffects

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

/-- Input-only values read through each actual Tensor's physicalIndex and
storage object. No ABI address or execution appears in this specification. -/
def actualInputs {A} [Inhabited A] (c : Shape) (t : Tensors) (before : Cells A) : Inputs A :=
  ⟨fun h r d => logical before t.q (c.mainOffset h r d),
   fun h r d => logical before t.k (c.mainOffset h r d),
   fun h r d => logical before t.v (c.mainOffset h r d)⟩

namespace InputBinding

def Agree (c : Shape) (head : Nat) (xs ys : Inputs A) : Prop :=
  ∀ r d, r<c.T → d<c.D → xs.q head r d=ys.q head r d ∧
    xs.k head r d=ys.k head r d ∧ xs.v head r d=ys.v head r d

theorem actual {A} [Inhabited A] (c : Shape) (abi : ABI c) (t : Tensors)
    (b : TensorBinding c abi t) (storeFlag ifcond : Bool)
    (admit : InputContract c (t.arguments storeFlag ifcond) b.pointers)
    (before : Cells A) (head : Nat) (hh : head<c.heads) :
    Agree c head (tensorInputs c abi before) (actualInputs c t before) := by
  intro r d hr hd
  exact inputs_logical c abi t b storeFlag ifcond admit before head r d hh hr hd

theorem masked_eq (ops : Ops A) (c : Shape) (x y : Nat → Nat → Nat → A) (head : Nat)
    (h : ∀ r d, r<c.T → d<c.D → x head r d=y head r d) (r d : Nat) :
    masked ops c x head r d=masked ops c y head r d := by
  unfold masked
  split_ifs with active
  · exact h r d active.1 active.2
  · rfl

theorem body (ops : Ops A) (c : Shape) (xs ys : Inputs A) (scale : A) (head chunk : Nat)
    (carry : Tile A) (ifcond : Bool) (h : Agree c head xs ys) :
    Body.evaluate ops c xs scale head chunk carry ifcond=
      Body.evaluate ops c ys scale head chunk carry ifcond := by
  have hq : Body.query ops c xs scale head chunk=Body.query ops c ys scale head chunk := by
    funext i d
    unfold Body.query
    rw [masked_eq ops c xs.q ys.q head (fun r d hr hd => (h r d hr hd).1)]
  have hk : Body.key ops c xs head chunk=Body.key ops c ys head chunk := by
    funext d i
    exact masked_eq ops c xs.k ys.k head (fun r d hr hd => (h r d hr hd).2.1) _ _
  have hv : Body.value ops c xs head chunk=Body.value ops c ys head chunk := by
    funext i d
    exact masked_eq ops c xs.v ys.v head (fun r d hr hd => (h r d hr hd).2.2) _ _
  simp only [Body.evaluate,hq,hk,hv]

theorem carry (ops : Ops A) (c : Shape) (xs ys : Inputs A) (scale : A) (head : Nat)
    (ifcond : Bool) (n : Nat) (h : Agree c head xs ys) :
    Reference.carry ops c xs scale head ifcond n=Reference.carry ops c ys scale head ifcond n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Reference.carry,Reference.carry,ih,body ops c xs ys scale head n _ ifcond h]

theorem output (ops : Ops A) (c : Shape) (xs ys : Inputs A) (scale : A) (head chunk : Nat)
    (ifcond : Bool) (h : Agree c head xs ys) :
    Reference.output ops c xs scale head chunk ifcond=Reference.output ops c ys scale head chunk ifcond := by
  rw [Reference.output,Reference.output,carry ops c xs ys scale head ifcond chunk h,
    body ops c xs ys scale head chunk _ ifcond h]

#axiomsClean actual
#axiomsClean output
#axiomsClean carry
end InputBinding
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
