import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelResult
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelRealModel

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.KernelRealResult
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
open Classical
set_option maxHeartbeats 2000000

theorem loaded (rest : Algebra ℝ) (p : Parameters) (s : State ℝ)
    (h : Fin p.height) (hh : h.val<p.heads) (values : Nat → ℝ)
    (reads : ∀ c,c<p.columns →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h.val c)).read .real=values c)
    (c : Fin p.width) :
    KernelResult.loaded (KernelRealModel.algebra rest) p s (h,c,PUnit.unit)=
      if c.val<p.columns then values c.val else 0 := by
  by_cases hc : c.val<p.columns
  · simp [KernelResult.loaded,KernelResult.active,KernelRealModel.algebra,hh,hc,reads _ hc]
  · simp [KernelResult.loaded,KernelResult.active,KernelRealModel.algebra,hh,hc]

theorem maximum (rest : Algebra ℝ) (p : Parameters) (s : State ℝ)
    (hp : 0<p.columns) (hb : p.columns≤p.width)
    (h : Fin p.height) (hh : h.val<p.heads) (values : Nat → ℝ)
    (reads : ∀ c,c<p.columns →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h.val c)).read .real=values c) :
    KernelResult.maximum (KernelRealModel.algebra rest) p s (h,PUnit.unit)=
      mathematicalMaximum values p.columns := by
  change KernelRealModel.maximum
    (fun c : Fin p.width => |KernelResult.loaded (KernelRealModel.algebra rest) p s (h,c,PUnit.unit)|)=_
  simp_rw [loaded rest p s h hh values reads]
  apply le_antisymm
  · apply KernelRealModel.maximum_le _ (lt_of_lt_of_le hp hb)
    intro c
    by_cases hc : c.val<p.columns
    · simpa [hc] using RowwiseQuantizationTriton.logicalMaximum_bound values p.columns c.val hc
    · simpa [hc] using RowwiseQuantizationTriton.logicalMaximum_nonnegative values p.columns
  · apply RowwiseQuantizationTriton.logicalMaximum_least
    · exact le_trans (abs_nonneg _) (KernelRealModel.maximum_bound
        (fun c : Fin p.width => |if c.val<p.columns then values c.val else 0|)
        ⟨0,lt_of_lt_of_le hp hb⟩)
    · intro c hc
      simpa [hc] using KernelRealModel.maximum_bound
        (fun j : Fin p.width => |if j.val<p.columns then values j.val else 0|)
        ⟨c,lt_of_lt_of_le hc hb⟩

theorem scale (rest : Algebra ℝ) (p : Parameters) (s : State ℝ)
    (hp : 0<p.columns) (hb : p.columns≤p.width)
    (h : Fin p.height) (hh : h.val<p.heads) (values : Nat → ℝ)
    (reads : ∀ c,c<p.columns →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h.val c)).read .real=values c) :
    KernelResult.scale (KernelRealModel.algebra rest) p s h=KernelRealModel.scale values p.columns := by
  change KernelResult.maximum (KernelRealModel.algebra rest) p s (h,PUnit.unit)/127=_
  rw [maximum rest p s hp hb h hh values reads]
  rfl

theorem code (rest : Algebra ℝ) (p : Parameters) (s : State ℝ)
    (hp : 0<p.columns) (hb : p.columns≤p.width)
    (h : Fin p.height) (hh : h.val<p.heads) (values : Nat → ℝ)
    (reads : ∀ c,c<p.columns →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h.val c)).read .real=values c)
    (c : Fin p.width) (hc : c.val<p.columns) :
    KernelResult.code (KernelRealModel.algebra rest) KernelRealModel.convert KernelRealModel.equal p s
      (h,c,PUnit.unit)=KernelRealModel.code values p.columns c.val := by
  simp only [KernelResult.code,KernelResult.denominator]
  rw [loaded rest p s h hh values reads c,if_pos hc,
    maximum rest p s hp hb h hh values reads,scale rest p s hp hb h hh values reads]
  by_cases hm : mathematicalMaximum values p.columns=0
  · have hz := (RowwiseQuantizationTriton.logicalMaximum_zero values p.columns).mp hm c.val hc
    simp [KernelRealModel.algebra,KernelRealModel.equal,KernelRealModel.convert,KernelRealModel.code,hm,hz]
  · simp [KernelRealModel.algebra,KernelRealModel.equal,KernelRealModel.convert,KernelRealModel.code,hm]

#axiomsClean loaded
#axiomsClean maximum
#axiomsClean scale
#axiomsClean code
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.KernelRealResult
