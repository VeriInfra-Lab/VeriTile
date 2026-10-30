import bench.tritonbench_g.token_attn_reduceV.Proofs.Bindings

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnReduceV
noncomputable section
variable {A : Type} [Inhabited A]

theorem products_logical (M : Algebra A) (p : Parameters) (B H : Nat) (counts : Fin B → Nat)
    (v : Inputs A p B H counts) (s : State A) (b : Fin B) (h : Fin H)
    (binding : CallInputs p B H counts v s b h) (n : Nat) :
    products M p s n = fun j : TileIndex [p.block,p.width] =>
      M.binary (some .fp32) .fp32 .mul
        (if n+j.1.val<counts b then probAt v b h (n+j.1.val) else padding M)
        (if n+j.1.val<counts b then valAt v b h (n+j.1.val) j.2.1 else padding M) := by
  funext j
  have nonzero : p.block≠0 := by have := j.1.isLt; omega
  have hp : probOffset p s j.1.val+n*p.probTokenStride = probOffset p s (n+j.1.val) := by
    unfold probOffset; ring
  have ht : gatherOffset p s j.1.val+n*p.requestTokenStride = gatherOffset p s (n+j.1.val) := by
    unfold gatherOffset; ring
  simp only [products,probability,value,gather,binding.length]
  by_cases active : n+j.1.val<counts b
  · simp only [active,if_true,hp,ht,binding.probability _ active nonzero,
      binding.value _ active nonzero j.2.1]
  · simp only [active,if_false]

theorem update_logical (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (B H : Nat) (counts : Fin B → Nat) (v : Inputs A p B H counts)
    (s : State A) (b : Fin B) (h : Fin H) (binding : CallInputs p B H counts v s b h) :
    update M R p s = orderedStep M R p (counts b) (probAt v b h) (valAt v b h) := by
  funext n a
  unfold update orderedStep
  rw [products_logical M p B H counts v s b h binding n]

theorem source_values (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (B H : Nat) (counts : Fin B → Nat) (v : Inputs A p B H counts)
    (s : State A) (b : Fin B) (h : Fin H) (binding : CallInputs p B H counts v s b h)
    (d : Fin p.width) :
    outputValues M R p s (d,PUnit.unit) =
      orderedRow M R p (counts b) (probAt v b h) (valAt v b h) d := by
  unfold outputValues accumulated orderedRow
  rw [update_logical M R p B H counts v s b h binding,binding.length]
  rfl

#axiomsClean source_values
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
