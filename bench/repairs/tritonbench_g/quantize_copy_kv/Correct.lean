import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.QuantizeCopyKv.Correct
open VeriTile.Triton FP.Structural KernelSource
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- The unchanged source stores signed INT8 codes and FP16 scales, with signed64 destination metadata.
Independent finite logical-row quantization. Both code and scale outputs
remain observed; zero extension is mathematical padding of the finite row. -/
def mathematical (p : Parameters) (count : Nat) (xs : LogicalView.Inputs p count) : LogicalView.Outputs p count where
  code := fun t h c =>
    let x := LogicalView.row xs t h
    if mathematicalMaximum x p.width=0 then 0 else
      let z := xs.values t h c/(mathematicalMaximum x p.width/127)
      (BitVec.ofInt 8 (if z<0 then -Int.floor (-z) else Int.floor z)).toInt
  scale := fun t h => mathematicalMaximum (LogicalView.row xs t h) p.width/127

specification kernel_correct (rest : Algebra ℝ) (p : Parameters) (count : Nat)
    (dest : Nat → Nat) (width : 0 < p.width) (covered : p.heads ≤ p.height)
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b) :
    Spec.Real (∀ version, LogicalView.quantizeIO rest p count dest version ⊨{LogicalView.Metadata dest} mathematical p count) := by
  intro version
  exact (LogicalReader.judgment_iff rest p count dest version).mpr
    (ReaderBefore.kernel_correct rest p count dest width covered unique version)

#axiomsClean kernel_correct
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.Correct
