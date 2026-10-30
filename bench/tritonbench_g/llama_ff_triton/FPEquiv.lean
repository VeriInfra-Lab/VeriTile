import bench.tritonbench_g.llama_ff_triton.PairIO

/-! The existing index-commuted candidate now uses the selected repaired
int64/four-mask source in both storage branches and the complete source grid.
Optional compiler-bound native observations remain in Misc/NativeCorrect.lean. -/
namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.FPEquiv
open VeriTile.Triton FP.Structural Execution
open scoped VeriTile.Spec
noncomputable section

abbrev original := @PairIO.original
abbrev candidate := @PairIO.candidate

/-- Exact full-grid FP equivalence for arbitrary selected scalar/K16 primitive
models and all admitted half/int8 payloads. Only signed integer output-index
multiplication is commuted; no floating rewrite or finite-word guard is used. -/
specification source_equivalence {α : Type} [Inhabited α]
    (a : Arithmetic α) (g : GateInstructions α) (p : Parameters) :
    original a g p ≡[[]] candidate a g p :=
  ArrayProgramIO.Implementation.of_equivalentOn (PairIO.variant a g p)
    (PairIO.equivalent a g p)

#axiomsClean source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.FPEquiv
