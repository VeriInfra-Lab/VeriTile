import bench.tritonbench_g.context_attn_llama.Proofs.Program

/-! The repaired direct FP32 kernel preserves this original typed floating
source. Only program IDs, int32 metadata/page loads and lane aranges are widened
to int64 in Python. Checked guards make every formed pointer and control
expression exact. Historical generic address helpers below are retained. -/
namespace VeriTile.Bench.Repairs.ContextAttnLlama
open VeriTile.Triton

abbrev Storage := VeriTile.Bench.TritonBenchG.ContextAttnLlama.Storage
abbrev Layout := VeriTile.Bench.TritonBenchG.ContextAttnLlama.Formula.Layout
abbrev Regions := VeriTile.Bench.TritonBenchG.ContextAttnLlama.Program.Regions

/-- Original typed FP32 source used by the direct checked call. -/
noncomputable def kernel (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.ContextAttnLlama.Program.kernel p r storage Bool.false scale

/-- Int64 two's-complement interpretation. Only active addresses are dereferenced. -/
def signed64 (n : Int) : Int :=
  let u := n % 18446744073709551616
  if u < 9223372036854775808 then u else u - 18446744073709551616

theorem signed64_exact (n : Int) (hn : 0 ≤ n) (hb : n < 9223372036854775808) :
    signed64 n = n := by
  have hw : n < 18446744073709551616 := by omega
  simp [signed64,Int.emod_eq_of_lt hn hw,hb]

/-- Finite tensor allocations already bound active element addresses. This is
not an additional small-index restriction: the bound is the signed size_t
allocation limit, and includes the 2^31 boundary exhibited by the CPU probe. -/
theorem address64_from_allocation (off extent : Nat)
    (hb : off < extent) (he : extent ≤ 9223372036854775808) :
    signed64 (off : Int) = off := by
  apply signed64_exact <;> omega

/-- Every nonnegative addend and partial sum is bounded by the full address;
thus widening before each source product avoids the original int32 overflow. -/
theorem partial_addresses_fit (a b c extent : Nat)
    (hb : a+b+c < extent) (he : extent ≤ 9223372036854775808) :
    a < 9223372036854775808 ∧ b < 9223372036854775808 ∧
    c < 9223372036854775808 ∧ a+b < 9223372036854775808 ∧
    a+b+c < 9223372036854775808 := by omega

end VeriTile.Bench.Repairs.ContextAttnLlama
