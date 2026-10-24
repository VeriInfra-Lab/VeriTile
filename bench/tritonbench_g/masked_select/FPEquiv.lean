import bench.tritonbench_g.masked_select.Kernels
import bench.tritonbench_g.masked_select.Proofs.HostComposition

/-! Source-selected FP32 storage, Bool mask and inclusive Int64 cumsum. All
2^32 payload patterns are retained exactly, including NaNs and signed zeros.
There is no payload arithmetic or numerical assumption. The same kernel as
Correct.lean runs for every trial and the selected configuration. -/
namespace VeriTile.Bench.TritonBenchG.MaskedSelect
open VeriTile.Triton FP.Structural TypedCopy Whole
noncomputable section

/-- Full returned FP32 tensor and finite source correspondence. The domain
0 < broadcast.numel < 2^31 covers every padded source lane, including inactive
ones. `NativeBuffers` fixes byte widths 4/1/8/4, alignment, 64-bit allocation
bounds and fresh output; it names the four actual contiguous source buffers.
`ScratchContract` permits only fresh producer-window scratch and preserves all
pre-existing cells. Kernel execution is derived, never a runtime callback. -/
specification masked_select_fp32 (rest : Algebra (BitVec 32)) (p : Parameters)
    (input : Tensor (BitVec 32)) (mask : Tensor Bool) (shape : List Nat)
    (trials : List LaunchPlan) (selected : LaunchPlan) (s : State (BitVec 32))
    (allocatedBefore owned : RegionName → Nat → Prop)
    (broadcast : broadcastShape input.shape mask.shape = some shape)
    (sameDevice : input.device = mask.device)
    (sizeDomain : 0 < shape.prod ∧ shape.prod < 2^31)
    (producerSep : ProducerSeparate p) (outputSep : Separate p)
    (ownership : ProducerOwnership { p with size := shape.prod } s
      (broadcastValues input shape) (broadcastValues mask shape) owned)
    (scratch : ScratchContract { p with size := shape.prod }
      (stableResult input mask shape).length allocatedBefore owned)
    (plansValid : ∀ plan ∈ selected :: trials, ValidPlan { p with size := shape.prod } plan)
    (buffers : NativeBuffers shape.prod
      (StableSelection.count (broadcastValues mask shape) shape.prod)) :
    ∃ t, runHost rest p input mask trials selected s =
        .ok (returned p input.device (stableResult input mask shape).length, t) ∧
      (∀ k, k < (stableResult input mask shape).length →
        some (t.mem p.output k) = ((stableResult input mask shape)[k]?).map (Cell.mk .real)) ∧
      (∀ r o, ¬ owned r o → r ≠ p.output ∨ (stableResult input mask shape).length ≤ o →
        t.mem r o = s.mem r o) ∧
      (∀ r o, allocatedBefore r o → t.mem r o = s.mem r o) ∧
      (∀ plan ∈ selected :: trials, NativePlanSafe { p with size := shape.prod } plan) ∧
      NativeAccessSafe (broadcastValues mask shape) shape.prod buffers := by
  obtain ⟨t, he, hv, hf, _⟩ := host_complete rest p input mask shape trials selected s owned
    broadcast sameDevice sizeDomain.1 producerSep outputSep ownership (plansValid selected (by simp))
  exact ⟨t, he, hv, hf, existing_memory_preserved _ _ allocatedBefore owned s t scratch hf,
    fun plan hp => plan_native _ plan (plansValid plan hp) sizeDomain.2,
    native_access_safe _ _ sizeDomain.2 buffers⟩

/-- Return metadata retains source FP32 storage and the input CUDA device. -/
theorem fp32_return_metadata (p : Parameters) (device length : Nat) :
    (returned p device length).storage = .fp32 ∧
      (returned p device length).shape = [length] ∧
      (returned p device length).device = device := ⟨rfl, rfl, rfl⟩

-- Structural raw-copy correctness uses no FP rewrite/derivation object.
-- Its full theorem dependency audit below contains only standard Lean axioms.
#axiomsClean masked_select_fp32
#axiomsClean fp32_return_metadata
end
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.MaskedSelect
