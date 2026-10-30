import bench.tritonbench_g.quantize_copy_kv.Proofs.Interface
import bench.tritonbench_g.quantize_copy_kv.Proofs.RawReal

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.Interface
open VeriTile.Triton FP.Structural KernelSource
noncomputable section

theorem real_implements (rest : Algebra ℝ) (p : Parameters) (count : Nat)
    (dest : Nat → Nat) (version : Bool) (width : 0 < p.width) (covered : p.heads ≤ p.height)
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b) :
    (programIO (KernelRealModel.algebra rest) KernelRealModel.convert KernelRealModel.equal
      p count dest version).ImplementsOn (metadataMatches p count dest)
      (fun values => result
        (fun t h c => KernelRealModel.code (payload values t h) p.width c)
        (fun t h => KernelRealModel.scale (payload values t h) p.width)) := by
  apply implements _ _ _ p count dest version width
  intro values s domain binding
  exact RawReal.kernel_correct rest p count dest width covered unique (payload values) s
    (metadata_binding _ _ _ p count dest version values domain s binding)
    (payload_binding _ _ _ p count dest version values s binding)

#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.Interface
