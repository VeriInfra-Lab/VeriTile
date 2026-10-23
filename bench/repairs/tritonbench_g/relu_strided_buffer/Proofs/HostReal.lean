import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Host
import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeReal

namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer
open HostPlan SignedView SignedGrid Whole
noncomputable section

/-- Nonempty checked signed views: exact returned out0, max(0,x) for every
logical output, entry-relative input/exterior frames and all active bounds. -/
specification relu_strided_buffer_checked_real (rest : Algebra ℝ) (x y : ViewMetadata)
    (admitted : checkInputs x y = .ok ()) (api : AllocationAPI x y)
    (s : State ℝ) (xs : Nat → ℝ)
    (values : ∀ i, i < x.n → s.mem x.buffer.region ((x.buffer.view x.stride).address i) = .mk .fp32 (xs i)) :
    VeriTile.Spec.Real (∃ final,
      runChecked (realBackend rest) x y Bool.true s = .ok (.returned y.buffer final) ∧
      (∀ i, i < x.n → final.mem y.buffer.region ((y.buffer.view y.stride).address i) =
        Cell.mk .fp32 (realResult xs i)) ∧
      (∀ i, i < x.n → final.mem x.buffer.region ((x.buffer.view x.stride).address i) =
        s.mem x.buffer.region ((x.buffer.view x.stride).address i)) ∧
      (∀ r a, (r ≠ y.buffer.region ∨ ∀ i, i < x.n → a ≠ (y.buffer.view y.stride).address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < x.n → (x.buffer.view x.stride).address i < x.capacity ∧
        (y.buffer.view y.stride).address i < y.capacity)) := by
  have h := (checkInputs_iff _ _).mp admitted
  have outputShape : y.buffer.shape = [x.n] := h.output.rank.1.trans (congrArg List.singleton h.shape.symm)
  have outputValid : (y.buffer.view y.stride).Valid x.n y.capacity := h.shape.symm ▸ ViewContract.valid y h.output
  obtain ⟨final,ran,result,inputFrame,frame,bounds⟩ := real_complete rest x.buffer y.buffer
    h.input.extent.1 x.stride y.stride h.input.rank.1 outputShape h.input.rank.2 h.output.rank.2
    (ViewContract.valid x h.input) outputValid h.outputStride (input_output_footprints x y h api) s xs values
  exact ⟨final,(checked_eq_source _ x y admitted _ s).trans (congrArg Except.ok ran),
    result,inputFrame,frame,bounds⟩
#axiomsClean relu_strided_buffer_checked_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReluStridedBuffer
