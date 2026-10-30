import bench.tritonbench_g.quantize_global.IO
import bench.tritonbench_g.quantize_global.Proofs.IOCorrect

/-! The complete selected source grid implements independent signed INT8
nearest-even quantization. The absolute maximum is an explicit logical input;
its scale condition is independent of execution. -/
namespace QuantizeGlobalRepair.ReaderBefore
open VeriTile VeriTile.Triton FP.Structural QuantizeIO
noncomputable section

theorem kernel_grid_correct (rest : Algebra ℝ) (p : Parameters) (sep : p.Separated)
    (trials : Nat → List Config)
    (nonempty : ∀ k, k < segmentCount p.n → trials k ≠ []) :
    Spec.Real ((io rest Real.arithmetic p trials).ImplementsOn (domain rest p trials)
      (target rest Real.arithmetic p trials (fun xs maximum i =>
        if maximum=0 then 0 else QuantizationReal.nearestEven (127*xs i/maximum)))) :=
  IOCorrect.real rest p sep trials nonempty

end
end QuantizeGlobalRepair.ReaderBefore
