import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.NativeProjection
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Native
open Whole
noncomputable section

/-- Exact compiler syntax and shape binding for an actual specialization.
Constructing this relation is not an artifact certificate. Native use also
requires the external admission checker and the documented ordinary compiler
contract; no whole-dot/reduction/value/run equality occurs in this relation. -/
structure Binding (descriptor : Descriptor) (c : Config) (selected : Descriptor) : Prop where
  admissible : descriptor.Admissible c
  selectedSyntax : selected=descriptor

/-- The Real projection is used only to relate the mathematical scale. Native
scale operations consume `scaleBits` itself and preserve both signed zeros. -/
def ScaleBinding (runtime : Runtime) (c : Config) : Prop :=
  ∃ q : Rat,(VeriTile.Triton.FP.Value.mk runtime.scaleBits : VeriTile.Triton.FP.Value .fp32).toRat?=some q ∧
    (q : ℝ)=c.scale

def compilerBoundary : String :=
  "Actual-specialization admission; exact shape-only ordered syntax; packed MMA and shared fragment routing; predicates; synchronization; allocator and runtime; scalar and fixed-K8 numerical leaves only"

theorem binding_history (d selected : Descriptor) (c : Config) (h : Binding d c selected)
    (xs : Nat → Graph) : selected.historySum xs=d.historySum xs := by rw [h.selectedSyntax]
theorem binding_correction (d selected : Descriptor) (c : Config) (h : Binding d c selected)
    (xs : Nat → Graph) : selected.correctionSum xs=d.correctionSum xs := by rw [h.selectedSyntax]

#axiomsClean binding_history
#axiomsClean binding_correction
#specNonCircular Binding avoiding [VeriTile.Triton.FP.Structural.exec,Typed.invoke,referenceQ,referenceK,referenceG]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Native
