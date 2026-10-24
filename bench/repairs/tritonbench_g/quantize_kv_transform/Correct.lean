import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.QuantizeKvTransform.Correct
open VeriTile.Triton FP.Structural KernelSource
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- The unchanged source stores signed INT8 codes and FP32 scales, with signed32 destination metadata and the complete supplied token permutation.
Independent finite logical-row quantization. Both code and scale outputs
remain observed; zero extension is mathematical padding of the finite row. -/
def mathematical (p : Parameters) (count : Nat) (xs : LogicalView.Inputs p count) : LogicalView.Outputs p count where
  code := fun t h c =>
    let x := LogicalView.row xs t h
    if mathematicalMaximum x p.columns=0 then 0 else
      let z := xs.values t h c/(mathematicalMaximum x p.columns/127)
      (BitVec.ofInt 8 (if z<0 then -Int.floor (-z) else Int.floor z)).toInt
  scale := fun t h => mathematicalMaximum (LogicalView.row xs t h) p.columns/127

specification kernel_real (rest : Algebra ℝ) (p : Parameters)
    (positive : 0 < p.columns) (headCover : p.heads ≤ p.height) (colCover : p.columns ≤ p.width)
    (count : Nat) (order : List Nat) (coverage : order.Perm (List.range count)) :
    Spec.Real (LogicalView.quantizeIO rest p count order ⊨{LogicalView.Destinations} mathematical p count) :=
  (LogicalReader.judgment_iff rest p count order).mpr
    (ReaderBefore.kernel_real rest p positive headCover colCover count order coverage)

#axiomsClean kernel_real
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.Correct
