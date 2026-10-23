import bench.repairs.tritonbench_g.kldiv_compute.revision002.Kernels
import bench.repairs.tritonbench_g.kldiv_compute.revision002.Proofs.Real

namespace VeriTile.Bench.Repairs.KldivCompute.Revision002.Correct
open VeriTile.Triton FP.Structural Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor World)
noncomputable section

/-- Default FP32 source; arbitrary CUDA input views with y.numel ≥ x.numel including
storage offsets, zero strides, aliases and lazy-negative views. Guards and
ordinary failures are executed in Host.run. No positivity is required for the
independent total Real pointwise formula. No Real=FP claim is made. -/
theorem same_actual_call_real {α σ : Type} [Inhabited α] (neg : α → α) (M : Algebra α)
    (rt : Runtime α σ) (lib : Library neg rt) (shadow : Math.Shadow rt)
    (x y output : Tensor) (initial final : World α σ)
    (_precisionX : x.dtype="torch.float32") (_precisionY : y.dtype="torch.float32")
    (hx : initial.live x.region) (hy : initial.live y.region)
    (returned : Host.run M rt x y initial=.ok output final)
    (before : Math.RealWorld) (live : before.live=initial.live) :
    ∃ e : ReturnedData neg M rt x y initial output final,
      ∃ after, Nonempty (Math.SameCallReal e before after) := by
  have actual := whole neg M rt lib x y initial hx hy
  rw [returned] at actual
  obtain ⟨e⟩ := actual
  obtain ⟨after,trace⟩ := Math.same_call neg M rt lib shadow x y output initial final e before live
    (live.symm ▸ hx) (live.symm ▸ hy)
  exact ⟨e,after,trace⟩

/-- Whole Real execution also stands independently, using the same typed
source, full grid and returned object with arbitrary Real initial memory. -/
theorem whole_real {σ : Type} (rt : Runtime ℝ σ) (lib : Library Neg.neg rt)
    (x y : Tensor) (initial : World ℝ σ) (hx : initial.live x.region) (hy : initial.live y.region) :
    Ensures (fun output final => Nonempty
      (ReturnedData Neg.neg Math.algebra rt x y initial output final))
      (Host.run Math.algebra rt x y initial) := whole _ _ _ lib _ _ _ hx hy

#axiomsClean same_actual_call_real
#axiomsClean whole_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Revision002.Correct
