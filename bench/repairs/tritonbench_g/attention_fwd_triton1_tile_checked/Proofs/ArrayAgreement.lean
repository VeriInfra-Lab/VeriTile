import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.Projection

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section
namespace ArrayAgreement

def Agree (c : Shape) (head : Nat) (xs ys : Inputs A) : Prop :=
  ∀ r d, r<c.T → d<c.D → xs.q head r d=ys.q head r d ∧
    xs.k head r d=ys.k head r d ∧ xs.v head r d=ys.v head r d

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

#axiomsClean output
#axiomsClean carry
end ArrayAgreement
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
