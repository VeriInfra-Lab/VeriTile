import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Preparation

namespace VeriTile.Bench.Repairs.LayerNormLiger.Equivalence
open VeriTile.Triton FP.Structural VeriTile.Torch
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (softwareSum)
noncomputable section

/-- Actual FP32 word result, including all stores, saved statistics, all SM
partials, real Torch axis sums/casts and returned descriptors. Native scalar
conformance remains local: no whole-output or reduction equality oracle. -/
specification layer_norm_source_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (literal : ℝ → FP.Value .fp32)
    (rsqrt : FP.Value .fp32 → FP.Value .fp32) (s : State (FP.Value .fp32))
    (x w b dy : TensorMetadata) (eps : Contract.Epsilon) (r : Wrapper.Inputs) (a : Wrapper.Allocations)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (h : Contract.InputContract x w b eps) (g : Contract.GradientContract dy x)
    (hp : 0 < programs) (storage : Wrapper.Storage (Contract.rows x) (Contract.width x) programs) :
    NativeIndex.Safe (Contract.rows x) (Contract.width x) programs (Settings.block (Contract.width x)) ∧
    Wrapper.checked (Forward.software rest cfg literal rsqrt) (softwareSum cfg) (Host.scalarFP32 cfg)
      x w b dy eps r a programs pw pb s =
      some (Host.finish (Host.scalarFP32 cfg) (Wrapper.backward r a x programs) programs pw pb
        (a.region 6) (a.region 7) dy.shape
        (Grid.reference (ClosedFP.backwardReference cfg literal (Wrapper.backward r a x programs))
          (Grid.reference (ClosedFP.forwardReference cfg literal rsqrt (Wrapper.forward r a x eps)) s
            (Contract.rows x)) programs)) := by
  refine ⟨Wrapper.native x w b eps programs h hp storage, ?_⟩
  rw [Wrapper.checked_eq _ _ _ _ _ _ _ _ _ _ _ _ _ _ h g hp]
  exact PublicReader.returned_software_bits rest cfg literal rsqrt _ _ _ _ _ _ _ _ _ s

/-- Selected complete reduction trees and local FMA contraction decisions
remain explicit in this source-preserving interpretation. -/
theorem layer_norm_selected_plan (P : NativePlan.Primitives) (contract : NativePlan.Contraction)
    (trees : NativePlan.Trees) (s : State NativePlan.Expr)
    (x w b dy : TensorMetadata) (eps : Contract.Epsilon) (r : Wrapper.Inputs) (a : Wrapper.Allocations)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (h : Contract.InputContract x w b eps) (g : Contract.GradientContract dy x) (hp : 0 < programs) :
    (Wrapper.checked (NativePlan.algebra contract trees) (NativePlan.sums contract trees)
      (NativePlan.hostScalar contract) x w b dy eps r a programs pw pb s).map
      (fun out => fun q o => NativePlan.evalCell P (out.state.mem q o)) =
      some (fun q o => NativePlan.evalCell P
        ((Host.finish (NativePlan.hostScalar contract) (Wrapper.backward r a x programs) programs pw pb
          (a.region 6) (a.region 7) dy.shape
          (Grid.reference (Backward.reference (NativePlan.algebra contract trees) (NativePlan.sums contract trees)
              (Wrapper.backward r a x programs))
            (Grid.reference (Forward.reference (NativePlan.algebra contract trees) (NativePlan.sums contract trees)
                (Wrapper.forward r a x eps)) s (Contract.rows x)) programs)).state.mem q o)) := by
  rw [Wrapper.checked_eq _ _ _ _ _ _ _ _ _ _ _ _ _ _ h g hp]
  exact PublicReader.returned_selected_plan_bits P contract trees _ _ _ _ _ _ _ _ _ s

abbrev python_checks_iff_preconditions := Contract.checkArguments_iff
abbrev source_forward_shape := Shape.forward_shape
abbrev source_backward_shape := Shape.backward_shape

#axiomsClean layer_norm_source_fp32
#axiomsClean layer_norm_selected_plan
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormLiger.Equivalence
