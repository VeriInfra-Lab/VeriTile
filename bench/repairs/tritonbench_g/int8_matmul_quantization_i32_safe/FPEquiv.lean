import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ProducerIndicesIO
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GemmIndicesIO

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.KernelOnly.FPEquiv
open VeriTile Triton FP.Structural KernelExecution
open scoped VeriTile.Spec
noncomputable section

/-- The producer always launches one row per CTA. Remove its redundant row
modulo while retaining the complete FP32 quantization graph and INT8/FP16
stores, every row order, actual safety, success, outputs and exterior frame. -/
specification producer_indices_equiv (f : KernelNativeModel.FloatingInstructions)
    (p : KernelProducerGrid.Geometry) (selection : Producer.Selection p) (covers : p.columns≤p.width)
    (rest : Algebra Word32) (order : Producer.Schedule p) :
    ProducerIndicesIO.original f p selection rest order ≡[[]]
      ProducerIndicesIO.optimized f p selection rest order :=
  ProgramIO.Implementation.of_equivalent
    (ProducerIndicesIO.variant (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8 p order)
    (ProducerIndicesIO.equivalent f p selection covers rest order)

/-- Wrapped GEMM row/column offsets make both scale-load masks redundant.
The complete typed-zero reset and both successful safe per-cell source traces
retain all signed32 operations, precision/casts, split orders and frames. -/
specification gemm_scale_masks_equiv (f : KernelNativeModel.FloatingInstructions)
    (tree : KernelScalarProfile.Tree) (rest : Algebra Word32) (plans : Plans)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c) :
    GemmIndicesIO.original f tree rest plans rowScale output s c columnMajor q ≡[[]]
      GemmIndicesIO.optimized f tree rest plans rowScale output s c columnMajor q :=
  ResetPairIO.Implementation.of_equivalentResetOnCells
    (GemmIndicesIO.variant (KernelScalarProfile.native f tree rest) signed32
      (KernelNativeModel.operations f).toOps rfl plans rowScale output s c columnMajor q)
    (GemmIndicesIO.equivalent f tree rest plans rowScale output s c columnMajor q)

#axiomsClean producer_indices_equiv
#axiomsClean gemm_scale_masks_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.KernelOnly.FPEquiv
