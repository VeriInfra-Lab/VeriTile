import bench.tritonbench_g.vector_addition.Proofs.Mathematical_MixedFP32Defined
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_MixedFP32Profiles

/-! Public concrete-software contracts for the 18 ordered integer/bool↔fp32
profiles. Totalized semantics expose every undefined-conversion policy. On
the active defined domain the independent memory result is policy-free.
Native numerical bindings, finite-width addressing, byte views and native
compiler/error/scheduling behavior are not established by these theorems. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.MixedFP32ValueCorrect
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters)
open IntegerSource (Canonical)
open MixedFP32 (Word)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.MixedFP32ValueCorrect
