import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.UnrolledPair

namespace VeriTile.Bench.Repairs.MultinomialSampling.FPEquiv
open VeriTile.Triton FP.Structural SourceValue
open scoped VeriTile.Spec
noncomputable section

def original {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α)
    (p : Parameters) (pid : Nat) := UnrolledPair.original rest P p pid

def unrolled {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α)
    (p : Parameters) (pid : Nat) := UnrolledPair.candidate rest P p pid

/-- Complete source token loop versus constexpr block-loop unrolling. Every
original token block, scan/max plan, FP32 operation, Philox10 call, signed64
label and offset conversion, mask and clamped label address is retained.
Undefined inactive-load inputs and the original signed64 seed domain remain. -/
specification static_unroll_equiv {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α)
    (p : Parameters) (pid : Nat) : original rest P p pid ≡[[]] unrolled rest P p pid := by
  exact VeriTile.Spec.FloatingPoint.ofStructural rfl rfl (UnrolledPair.equivalent rest P p pid)

#print_fp_assumptions static_unroll_equiv
#axiomsClean static_unroll_equiv
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.FPEquiv
