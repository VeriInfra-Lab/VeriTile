import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.QuantizeKvCopy.Correct
open VeriTile.Triton FP.Structural Kernel KernelModels
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- The unchanged source stores signed INT8 codes and FP32 scales. Its complete schedule retains n flattened token/head program IDs, decoded by the original IO.
Independent finite logical-row quantization. Both code and scale outputs
remain observed; zero extension is mathematical padding of the finite row. -/
def formula (p : Parameters) (n : Nat) (xs : LogicalView.Inputs p n) : LogicalView.Outputs p n where
  code := fun t h c =>
    let x := LogicalView.row xs t h
    if mathematicalMaximum x 8=0 then 0 else
      let z := xs.values t h c/(mathematicalMaximum x 8/127)
      (BitVec.ofInt 8 (if z<0 then -Int.floor (-z) else Int.floor z)).toInt
  scale := fun t h => mathematicalMaximum (LogicalView.row xs t h) 8/127

specification mathematical (p : Parameters) (n : Nat) (dest : Nat → Nat)
    (covers : p.groups ≤ p.block)
    (unique : ∀ a b, a<n → b<n →
      KernelGridResult.target p dest a = KernelGridResult.target p dest b → a=b) :
    Spec.Real (LogicalView.quantizeIO p n dest ⊨{LogicalView.Metadata dest} formula p n) :=
  (LogicalReader.judgment_iff p n dest).mpr (ReaderBefore.mathematical p n dest covers unique)

#axiomsClean mathematical
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.Correct
