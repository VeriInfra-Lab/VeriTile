import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Proofs.PairEquivalence

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
open scoped VeriTile.Spec
noncomputable section
abbrev original := @PairIO.original
abbrev blockEndFactored := @PairIO.blockEndFactored

/-- Complete actual FP32 programs: common prompt-offset factoring in the
integer minimum saves one scalar add per CTA. Full paged input domain,
arbitrary order/duplicate jobs, aliases, all typed final outputs and both
exterior frames are retained. No floating law or finite-word guard is used. -/
specification block_end_factored_equiv {α : Type} [Inhabited α]
    (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (order : List (GridIndex c.grid))
    (separate : c.regions.start≠c.regions.out ∧ c.regions.sequence≠c.regions.out ∧ c.regions.prompt≠c.regions.out) :
    original c M dots order ≡[[]] blockEndFactored c M dots order :=
  PairEquivalence.equivalent c M dots order separate
#axiomsClean block_end_factored_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.FPEquiv
