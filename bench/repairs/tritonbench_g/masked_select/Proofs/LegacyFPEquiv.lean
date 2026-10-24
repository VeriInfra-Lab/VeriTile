import bench.repairs.tritonbench_g.masked_select.Kernels
import bench.repairs.tritonbench_g.masked_select.Proofs.Host

/-! Complete guarded source result. The independent target is the stable
row-major filter after logical broadcasting. Existing whole proofs are reused;
no kernel result, execution-success callback or payload arithmetic is assumed. -/
namespace VeriTile.Bench.Repairs.MaskedSelect
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy
open VeriTile.Bench.TritonBenchG.MaskedSelect.Whole
open VeriTile.Bench.TritonBenchG.MaskedSelect
noncomputable section

specification masked_select_fp32 (rest : Algebra (BitVec 32)) (p : Parameters)
    (input : Tensor (BitVec 32)) (mask : Tensor Bool) (shape : List Nat)
    (trials : List LaunchPlan) (selected : LaunchPlan) (s : State (BitVec 32))
    (allocatedBefore owned : RegionName → Nat → Prop)
    (x m : VeriTile.Torch.TensorMetadata) (h : InputContract x m shape)
    (inputShape : input.shape = x.shape) (maskShape : mask.shape = m.shape)
    (inputDevice : input.device = x.device.index) (maskDevice : mask.device = m.device.index)
    (preparedPointer : Nat) (prepared : PreparedContract preparedPointer)

    (producerSep : ProducerSeparate p) (outputSep : Separate p)
    (ownership : ProducerOwnership { p with size := shape.prod } s
      (broadcastValues input shape) (broadcastValues mask shape) owned)
    (scratch : ScratchContract { p with size := shape.prod }
      (stableResult input mask shape).length allocatedBefore owned)
    (plansValid : ∀ plan ∈ selected :: trials, ValidPlan { p with size := shape.prod } plan)
    (buffers : NativeBuffers shape.prod
      (StableSelection.count (broadcastValues mask shape) shape.prod))
    (_pointerBinding : preparedPointer = buffers.input.base) :
    ∃ t, runChecked rest p x m input mask preparedPointer trials selected s =
        .ok (.ok (returned p input.device (stableResult input mask shape).length, t)) ∧
      (∀ k, k < (stableResult input mask shape).length →
        some (t.mem p.output k) = ((stableResult input mask shape)[k]?).map (Cell.mk .real)) ∧
      (∀ r o, ¬ owned r o → r ≠ p.output ∨ (stableResult input mask shape).length ≤ o →
        t.mem r o = s.mem r o) ∧
      (∀ r o, allocatedBefore r o → t.mem r o = s.mem r o) ∧
      (∀ plan ∈ selected :: trials, NativePlanSafe { p with size := shape.prod } plan) ∧
      NativeAccessSafe (broadcastValues mask shape) shape.prod buffers := by
  have broadcast : broadcastShape input.shape mask.shape = some shape := by
    rw [inputShape, maskShape]; exact h.broadcast
  have sameDevice : input.device = mask.device := by
    rw [inputDevice,maskDevice,h.device]
  obtain ⟨t, he, hv, hf, _⟩ := host_complete rest p input mask shape trials selected s owned
    broadcast sameDevice h.extent.1 producerSep outputSep ownership (plansValid selected (by simp))
  have checked := checked_eq_original rest p x m input mask shape h preparedPointer prepared
    inputShape maskShape trials selected s
  rw [he] at checked
  exact ⟨t, checked, hv, hf, existing_memory_preserved _ _ allocatedBefore owned s t scratch hf,
    fun plan hp => plan_native _ plan (plansValid plan hp) h.extent.2,
    native_access_safe _ _ h.extent.2 buffers⟩

#axiomsClean masked_select_fp32
end
end VeriTile.Bench.Repairs.MaskedSelect
