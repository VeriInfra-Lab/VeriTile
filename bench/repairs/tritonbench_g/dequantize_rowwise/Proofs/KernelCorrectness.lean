import bench.repairs.tritonbench_g.dequantize_rowwise.Kernels
import bench.tritonbench_g.dequantize_rowwise.Proofs.Projection

namespace VeriTile.Bench.Repairs.DequantizeRowwise.KernelCorrectness
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DequantizeRowwise
noncomputable section

/-- Independent rowwise dequantization formula, including negative weights
and zero or negative scales. -/
def dequantized (scale : ℝ) (weight : Int) : ℝ := scale * (weight : ℝ) / 127

theorem source_exec (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) (u : BlockState) :
    exec (source x scale out inv n block lanes).eraseDType.toAlgKernel u =
      exec (RealReference.dequantize_rowwise_kernel x scale out inv n block lanes).toAlgKernel u := by
  change exec ((if Bool.false then TypedSource.indexCommuted else TypedSource.original)
    x scale out inv n block lanes).eraseDType.toAlgKernel u = _
  simp only [exec, Projection.source_body]
  rfl

def Result (p : Parameters) (u v : BlockState) : Prop :=
  (∀ column, column < p.cols →
    v.readMem outRegion (u.pid * p.cols + column) =
      dequantized (u.readMem scaleRegion u.pid)
        (u.readMemValue .int xRegion (u.pid * p.cols + column))) ∧
  (∀ r o, (r ≠ outRegion ∨ ∀ column, column < p.cols → o ≠ u.pid * p.cols + column) →
    v.mem r o = u.mem r o)

/-- Complete single-program execution and regional memory frame. Input
regions are read independently, so their backing views may overlap. -/
theorem correct (p : Parameters) (u : BlockState) :
    ∃ v, exec (kernel p).eraseDType.toAlgKernel u = some v ∧ Result p u v := by
  obtain ⟨v, he, ho, hf⟩ := RealReference.dequantize_rowwise_kernel_region_run
    xRegion scaleRegion outRegion (1 / 127) (p.rows * p.cols) p.cols p.width u
    (fun j => u.readMemValue .int xRegion (u.pid * p.cols + j.val))
    (fun _ => u.readMem scaleRegion u.pid) (by intros; rfl) (by intros; rfl)
  refine ⟨v, (source_exec _ _ _ _ _ _ _ u).trans he, ?_, ?_⟩
  · intro column hc
    simpa only [dequantized, div_eq_mul_inv, one_mul] using
      ho ⟨column, lt_of_lt_of_le hc p.covers⟩ hc
  · intro r o miss
    exact hf r o (miss.imp id (fun h j hj => h j.val hj))

/-- The original scale load is unmasked, including when the row has no
active output lane. Its capacity is therefore an independent scalar bound. -/
theorem traceSafe (p : Parameters) (bounds : RegionBounds) (u : BlockState)
    (weights : ∀ j : Fin p.width, j.val < p.cols →
      u.pid * p.cols + j.val < bounds xRegion)
    (scale : u.pid < bounds scaleRegion)
    (output : ∀ j : Fin p.width, j.val < p.cols →
      u.pid * p.cols + j.val < bounds outRegion) :
    Kernel.TraceSafe bounds (kernel p).eraseDType.toAlgKernel u := by
  have safe := RealReference.dequantize_rowwise_kernel_traceSafe
    xRegion scaleRegion outRegion (1 / 127) (p.rows * p.cols) p.cols p.width
    bounds u weights scale output
  have body : (kernel p).eraseDType.toAlgKernel.body =
      (RealReference.dequantize_rowwise_kernel xRegion scaleRegion outRegion
        (1 / 127) (p.rows * p.cols) p.cols p.width).toAlgKernel.body := by
    change ((if Bool.false then TypedSource.indexCommuted else TypedSource.original)
      xRegion scaleRegion outRegion (1 / 127) (p.rows * p.cols) p.cols p.width).eraseDType.toAlgKernel.body = _
    rw [Projection.source_body]
    rfl
  simpa only [Kernel.TraceSafe, body] using safe

/-- Bind the typed INT8 row and the independently read FP32 scale to logical
inputs. Zero columns and read-only aliases retain the original regional scope. -/
theorem inputs_correct (p : Parameters) (u : BlockState)
    (weights : Fin p.width → Int) (scale : ℝ)
    (weightsRead : ∀ j : Fin p.width, j.val < p.cols →
      u.readMemValue .int xRegion (u.pid * p.cols + j.val) = weights j)
    (scaleRead : u.readMem scaleRegion u.pid = scale) :
    ∃ v, exec (kernel p).eraseDType.toAlgKernel u = some v ∧
      (∀ j : Fin p.width, j.val < p.cols →
        v.readMem outRegion (u.pid * p.cols + j.val) = dequantized scale (weights j)) ∧
      (∀ r o, (r ≠ outRegion ∨ ∀ j : Fin p.width, j.val < p.cols →
        o ≠ u.pid * p.cols + j.val) → v.mem r o = u.mem r o) := by
  obtain ⟨v, execution, values, frame⟩ := correct p u
  refine ⟨v, execution, ?_, ?_⟩
  · intro j active
    rw [values j.val active, scaleRead, weightsRead j active]
  · intro r o outside
    apply frame r o
    exact outside.imp id (fun outside j active =>
      outside ⟨j, lt_of_lt_of_le active p.covers⟩ active)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeRowwise.KernelCorrectness
