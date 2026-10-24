import bench.repairs.tritonbench_g.fused_activation.Correct
import bench.repairs.tritonbench_g.fused_activation.Proofs.InputBinding

namespace VeriTile.Bench.Repairs.FusedActivation.TensorBridge
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.FusedActivation
open Correct
noncomputable section

open VeriTile.Torch.FP32 in
/-- Compose checked Tensor storage observations with the actual kernel proof.
Bias and input use the same reader and may overlap at arbitrary offsets.
The conclusion contains execution, logical mathematical outputs, and the full
regional exterior frame; it is not merely a lemma about input reads. -/
theorem fused_activation_checked_views_correctness
    (read : Nat → Word) (decode : Word → ℝ) (x b input : Tensor)
    (lx : TensorLaws x) (lb : TensorLaws b) (li : TensorLaws input)
    (h : InputContract x.metadata b.metadata input.metadata)
    (a : Bool) (scratch : BlockState) :
    let s := InputBinding.bindReadViews read decode x b input lx lb li h scratch
    let N := x.metadata.numel
    let W := b.metadata.numel
    let B := block N
    Spec.Real (∃ final, exec (kernel "x" "bias" "input" W N B (1/2) a).toAlgKernel s = some final ∧
      (∀ j : Fin B, s.pid * B + j.val < N →
        final.readMem "x" (s.pid * B + j.val) = formula a
          (decode (logicalValue read x lx (s.pid * B + j.val)))
          (decode (logicalValue read b lb ((s.pid * B + j.val) % W)))
          (decode (logicalValue read input li (s.pid * B + j.val)))) ∧
      (∀ (r : RegionName) (o : Nat), (∀ j : Fin B, s.pid * B + j.val < N →
        r ≠ "x" ∨ o ≠ s.pid * B + j.val) → final.mem r o = s.mem r o)) := by
  obtain ⟨hx, hb, hi⟩ := InputBinding.bindReadViews_logical read decode x b input lx lb li h scratch
  let s := InputBinding.bindReadViews read decode x b input lx lb li h scratch
  let xs : Fin 3 → Fin (block x.metadata.numel) → ℝ := fun channel j =>
    match channel with
    | ⟨0, _⟩ => decode (logicalValue read x lx (s.pid * block x.metadata.numel + j.val))
    | ⟨1, _⟩ => decode (logicalValue read b lb ((s.pid * block x.metadata.numel + j.val) % b.metadata.numel))
    | ⟨_ + 2, _⟩ => decode (logicalValue read input li (s.pid * block x.metadata.numel + j.val))
  obtain ⟨final, he, hv, hf⟩ := GroupedMasked2DKernelIO.RegionalImplements.run
    (fused_activation_kernel_correctness "x" "bias" "input"
      b.metadata.numel x.metadata.numel (block x.metadata.numel) a) s xs (by
        intro channel j active
        fin_cases channel
        · exact hx _ active
        · exact hb _ (Nat.mod_lt _ ((h.2.2.2.2.2.2.2.2.2 (Nat.zero_lt_of_lt active)).1))
        · exact hi _ active)
  refine ⟨final, he, fun j hj => hv (⟨0, by decide⟩ : Fin 1) j hj, ?_⟩
  intro r o outside
  exact hf r o (fun _ j hj => outside j hj)

alias checked_tensor_reads := InputBinding.logical_reads
alias checked_tensor_pointers := InputBinding.actual_pointers

theorem python_checks_iff_preconditions (x b i : ArgumentMetadata) :
    checkInputs x b i = .ok () ↔ AdmittedArguments x b i := checkInputs_iff x b i

#axiomsClean fused_activation_checked_views_correctness
#axiomsClean checked_tensor_reads
#axiomsClean checked_tensor_pointers
#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedActivation.TensorBridge
