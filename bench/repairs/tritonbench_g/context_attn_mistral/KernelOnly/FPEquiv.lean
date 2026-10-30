import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Proofs.PairEquivalence

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
open scoped VeriTile.Spec
noncomputable section

abbrev original := @PairIO.original
abbrev blockStartReused := @PairIO.blockStartReused

/-- Complete FP32 sources on the original arbitrary-job array domain. Reusing
`block_start_loc` removes one repeated index multiply per CTA. Every floating
instruction, mask, conversion and dot accumulation order remains unchanged.
Both schedules succeed, are access-safe, agree at every active typed output,
and preserve their complete exterior frames, including aliases/repeated jobs. -/
specification block_start_reused_equiv {α : Type} [Inhabited α]
    (c : Host.Call) (o : FPBackend.Ops α) (jobs : List (GridIndex c.grid))
    (separate : c.regions.out≠c.regions.start ∧ c.regions.out≠c.regions.sequence) :
    original c o jobs ≡[[]] blockStartReused c o jobs :=
  PairEquivalence.equivalent c o jobs separate

#axiomsClean block_start_reused_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.FPEquiv
