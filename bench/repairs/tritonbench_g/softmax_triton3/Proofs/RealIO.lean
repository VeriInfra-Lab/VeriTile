import bench.repairs.tritonbench_g.softmax_triton3.IO
import bench.repairs.tritonbench_g.softmax_triton3.Proofs.KernelRealResult

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.RealIO
open VeriTile.Triton
noncomputable section
/-- Independent mathematical row formula; only the active logical columns
participate in the denominator. -/
def rowFormula (n B : Nat) (xs : Fin B → ℝ) (j : Fin B) : ℝ :=
  Real.exp (xs j) / ∑ k : Fin B, if k.val<n then Real.exp (xs k) else 0

/-- Every active row is executed safely and implements mathematical softmax;
all other memory cells retain their original values. -/
theorem unmasked (out inp : RegionName) (n B : Nat)
    (positive : 0<B) (_cover : n≤B) :
    Spec.Real (MaskedKernelIO₁.Implements (Correct.softmaxIO out inp n B) (rowFormula n B)) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · simpa only [Correct.softmaxIO] using
      RawReal.softmax_kernel_flattenOk out inp inp n n B Bool.false
  · intro bounds s hin hout _
    simpa only [Correct.softmaxIO] using
      RawReal.softmax_kernel_traceSafe out inp inp n n B
        positive bounds s hin hout
  · intro s xs hx
    obtain ⟨t,run,values,frame⟩ := RawReal.softmax_kernel_region_run
      out inp inp n n B positive s xs hx
    refine ⟨t,?_,fun j hj => ?_,fun r o miss _ => frame r o miss⟩
    · simpa only [Correct.softmaxIO] using run
    · exact (values j hj).trans
        (TritonBenchG.SoftmaxTriton3.softmaxSpec_none_eq_formula n B positive xs j hj)

/-- The optional mask is added to each logical logit before mathematical
softmax. The source uses the same row windows and active-column mask. -/
theorem masked (out inp mask : RegionName) (n B : Nat)
    (positive : 0<B) (_cover : n≤B) :
    Spec.Real (MaskedKernelIO₂.Implements (Correct.softmaxMaskedIO out inp mask n B)
      (fun xs ms => rowFormula n B (fun j => xs j+ms j))) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · simpa only [Correct.softmaxMaskedIO] using
      RawReal.softmax_kernel_flattenOk out inp mask n n B Bool.true
  · intro bounds s hin hm hout _
    simpa only [Correct.softmaxMaskedIO] using
      RawReal.softmax_kernel_masked_traceSafe out inp mask n n B
        positive bounds s hin hm hout
  · intro s xs ms hx hm
    obtain ⟨t,run,values,frame⟩ := RawReal.softmax_kernel_masked_region_run
      out inp mask n n B positive s xs ms hx hm
    refine ⟨t,?_,fun j hj => ?_,fun r o miss _ => frame r o miss⟩
    · simpa only [Correct.softmaxMaskedIO] using run
    · exact (values j hj).trans
        (TritonBenchG.SoftmaxTriton3.softmaxSpec_mask_eq_formula n B positive xs ms j hj)

#axiomsClean unmasked
#axiomsClean masked
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.RealIO
