import bench.repairs.tritonbench_g.context_attn_llama.Kernels
import bench.tritonbench_g.context_attn_llama.Proofs.Host

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Address
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnLlama
open Formula Program AddressBounds
noncomputable section

/-- The signed allocation-size ABI, not an int32-safe subset of valid views. -/
def Allocation64 (extent : RegionBounds) : Prop :=
  ∀ r, extent r ≤ 9223372036854775808

theorem inside_exact {a : Int} {extent : Nat}
    (h : inside a extent) (he : extent ≤ 9223372036854775808) : signed64 a = a := by
  apply signed64_exact a h.1
  have hcast : (extent : Int) ≤ 9223372036854775808 := by exact_mod_cast he
  exact lt_of_lt_of_le h.2 hcast

/-- Each actual source address is represented exactly after widening, for all
programs and reached lanes. Inactive lane addresses are never dereferenced. -/
theorem active_addresses_exact (p : Layout) (r : Regions) (l : Launch) (a : Inputs)
    (extent : RegionBounds) (h : Valid p r l a extent) (he : Allocation64 extent) :
    (∀ i : Fin p.BM, ∀ d : Fin p.D, (queryRow p l i.val : Int) < Formula.sequence p l a →
      signed64 (queryAddress p l a i d) = queryAddress p l a i d) ∧
    (∀ c, c < count p l a → ∀ j : Fin p.BN,
      ((c*p.BN+j.val : Nat) : Int) < Formula.limit p l a →
      signed64 (pageAddress p l a (c*p.BN+j.val)) = pageAddress p l a (c*p.BN+j.val)) ∧
    (∀ c, c < count p l a → ∀ j : Fin p.BN, ∀ d : Fin p.D,
      ((c*p.BN+j.val : Nat) : Int) < Formula.limit p l a →
      signed64 (keyAddress p l a (c*p.BN+j.val) d) = keyAddress p l a (c*p.BN+j.val) d) ∧
    (∀ c, c < count p l a → ∀ j : Fin p.BN, ∀ d : Fin p.D,
      ((c*p.BN+j.val : Nat) : Int) < Formula.limit p l a →
      signed64 (valueAddress p l a (c*p.BN+j.val) d) = valueAddress p l a (c*p.BN+j.val) d) ∧
    (∀ i : Fin p.BM, ∀ d : Fin p.D, (queryRow p l i.val : Int) < Formula.sequence p l a →
      signed64 (outputAddress p l a i d) = outputAddress p l a i d) := by
  exact ⟨fun i d hi => inside_exact (h.query i d hi) (he r.query),
    fun c hc j hj => inside_exact (h.pages c hc j hj) (he r.pages),
    fun c hc j d hj => inside_exact (h.key c hc j d hj) (he r.key),
    fun c hc j d hj => inside_exact (h.value c hc j d hj) (he r.value),
    fun i d hi => inside_exact (h.output i d hi) (he r.out)⟩

/-- Both native block configurations safely widen the program/row/prompt
control expressions from their original int32 ABI values before arithmetic. -/
theorem control_expression_exact (pid prompt bm lane : Nat)
    (hp : pid < 2147483648) (hc : prompt < 2147483648)
    (hm : bm = 64 ∨ bm = 128) (hl : lane < bm) :
    signed64 ((pid*bm+lane+prompt : Nat) : Int) = pid*bm+lane+prompt ∧
    signed64 ((pid*bm+bm+prompt : Nat) : Int) = pid*bm+bm+prompt := by
  rcases hm with hm | hm <;> subst bm <;> constructor <;>
    apply signed64_exact <;> omega

end
end VeriTile.Bench.Repairs.ContextAttnLlama.Address
