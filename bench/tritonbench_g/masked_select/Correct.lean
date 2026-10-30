import bench.tritonbench_g.masked_select.Kernels
import bench.tritonbench_g.masked_select.Proofs.HostComposition

/-! Real is the independent mathematical specification: broadcast, then retain
selected row-major logical elements in their original order. The shared source
only copies payloads, so its real projection needs no numerical laws. Original
shape assertion and empty-prefix failures are public below. -/
namespace VeriTile.Bench.TritonBenchG.MaskedSelect
open VeriTile.Triton FP.Structural TypedCopy Whole
noncomputable section

theorem shared_real_source (p : Parameters) :
    (Whole.kernel p).eraseDType.toAlgKernel.body =
      (masked_select_kernel p.input p.selection p.prefixSum p.output p.size p.width).toAlgKernel.body :=
  TypedCopy.floating_source_projection .fp32 (by decide) p

/-- Complete logical host call, including varying-width autotune trials and
the selected actual kernel grid. Bool mask is an API condition. Ordinary Torch
contracts supply logical tensor views, successful primitive calls and the
explicit fresh-scratch/identity-contiguous ownership facts. -/
specification masked_select_real (rest : Algebra ℝ) (p : Parameters)
    (input : Tensor ℝ) (mask : Tensor Bool) (shape : List Nat)
    (trials : List LaunchPlan) (selected : LaunchPlan) (s : State ℝ)
    (allocatedBefore owned : RegionName → Nat → Prop)
    (broadcast : broadcastShape input.shape mask.shape = some shape)
    (sameDevice : input.device = mask.device) (nonempty : 0 < shape.prod)
    (producerSep : ProducerSeparate p) (outputSep : Separate p)
    (ownership : ProducerOwnership { p with size := shape.prod } s
      (broadcastValues input shape) (broadcastValues mask shape) owned)
    (scratch : ScratchContract { p with size := shape.prod }
      (stableResult input mask shape).length allocatedBefore owned)
    (plansValid : ∀ plan ∈ selected :: trials, ValidPlan { p with size := shape.prod } plan) :
    VeriTile.Spec.Real (∃ t, runHost rest p input mask trials selected s =
        .ok (returned p input.device (stableResult input mask shape).length, t) ∧
      (∀ k, k < (stableResult input mask shape).length →
        some (t.mem p.output k) = ((stableResult input mask shape)[k]?).map (Cell.mk .real)) ∧
      (∀ r o, ¬ owned r o → r ≠ p.output ∨ (stableResult input mask shape).length ≤ o →
        t.mem r o = s.mem r o) ∧
      (∀ r o, allocatedBefore r o → t.mem r o = s.mem r o)) := by
  obtain ⟨t, he, hv, hf, _⟩ := host_complete rest p input mask shape trials selected s owned
    broadcast sameDevice nonempty producerSep outputSep ownership (plansValid selected (by simp))
  exact ⟨t, he, hv, hf, existing_memory_preserved _ _ allocatedBefore owned s t scratch hf⟩

export Whole (host_shape_failure host_empty_failure all_false_output)

#specNonCircular Whole.stableResult avoiding [Whole.runHost, Whole.runGrid, Whole.runLaunches, Whole.kernel]
#axiomsClean shared_real_source
#axiomsClean masked_select_real
end
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.MaskedSelect
