import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Proofs.PairEquivalence

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section
abbrev original := @PairIO.original
abbrev blockStartReused := @PairIO.blockStartReused

/-- Complete selected FP32 grids. Reuse the previously computed scalar block
origin, removing one repeated index multiplication per CTA. The original masks,
exp, conversions, normalization, dot/FMA structure and all-word domain remain.
Both source schedules are safe and succeed, every active typed output agrees,
and both complete exterior frames are retained, including floating aliases. -/
specification block_start_reused_equiv {α : Type} [Inhabited α]
    (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (startLabel : r.out≠r.start) (lengthLabel : r.out≠r.sequence) :
    original c r o ≡[[]] blockStartReused c r o :=
  PairEquivalence.equivalent c r o startLabel lengthLabel
#axiomsClean block_start_reused_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.FPEquiv
