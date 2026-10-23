import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelResult
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelRealModel

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.KernelRealResult
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
open Classical
set_option maxHeartbeats 1500000

theorem loaded (rest : Algebra ℝ) (p : Parameters) (s : State ℝ)
    (h : Fin p.height) (hh : h.val < p.heads) (values : Nat → ℝ)
    (reads : ∀ c, c < p.width →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h.val c)).read .real = values c)
    (c : Fin p.width) :
    KernelResult.loaded (KernelRealModel.algebra rest) p s (h, c, PUnit.unit) = values c.val := by
  simp [KernelResult.loaded, KernelRealModel.algebra, hh, reads c.val c.isLt]

theorem maximum (rest : Algebra ℝ) (p : Parameters) (s : State ℝ) (hw : 0 < p.width)
    (h : Fin p.height) (hh : h.val < p.heads) (values : Nat → ℝ)
    (reads : ∀ c, c < p.width →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h.val c)).read .real = values c) :
    KernelResult.maximum (KernelRealModel.algebra rest) p s (h, PUnit.unit) =
      mathematicalMaximum values p.width := by
  change KernelRealModel.maximum
    (fun c : Fin p.width => |KernelResult.loaded (KernelRealModel.algebra rest) p s (h, c, PUnit.unit)|) = _
  have values_eq :
      (fun c : Fin p.width => |KernelResult.loaded (KernelRealModel.algebra rest) p s (h, c, PUnit.unit)|) =
      (fun c : Fin p.width => |values c.val|) := by
    funext c
    rw [loaded rest p s h hh values reads c]
  rw [values_eq, KernelRealModel.maximum_abs hw]

theorem scale (rest : Algebra ℝ) (p : Parameters) (s : State ℝ) (hw : 0 < p.width)
    (h : Fin p.height) (hh : h.val < p.heads) (values : Nat → ℝ)
    (reads : ∀ c, c < p.width →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h.val c)).read .real = values c) :
    KernelResult.scale (KernelRealModel.algebra rest) p s h = KernelRealModel.scale values p.width := by
  change KernelResult.maximum (KernelRealModel.algebra rest) p s (h, PUnit.unit) / 127 = _
  rw [maximum rest p s hw h hh values reads]
  rfl

theorem code (rest : Algebra ℝ) (p : Parameters) (s : State ℝ) (hw : 0 < p.width)
    (h : Fin p.height) (hh : h.val < p.heads) (values : Nat → ℝ)
    (reads : ∀ c, c < p.width →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h.val c)).read .real = values c)
    (c : Fin p.width) :
    KernelResult.code (KernelRealModel.algebra rest) KernelRealModel.convert KernelRealModel.equal p s
      (h, c, PUnit.unit) = KernelRealModel.code values p.width c.val := by
  simp only [KernelResult.code, KernelResult.denominator]
  rw [loaded rest p s h hh values reads c,
    maximum rest p s hw h hh values reads, scale rest p s hw h hh values reads]
  by_cases hm : mathematicalMaximum values p.width = 0
  · have hz := (RowwiseQuantizationTriton.logicalMaximum_zero values p.width).mp hm c.val c.isLt
    simp [KernelRealModel.algebra, KernelRealModel.equal, KernelRealModel.convert,
      KernelRealModel.code, hm, hz]
  · simp [KernelRealModel.algebra, KernelRealModel.equal, KernelRealModel.convert,
      KernelRealModel.code, hm]

#axiomsClean loaded
#axiomsClean maximum
#axiomsClean scale
#axiomsClean code
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.KernelRealResult
