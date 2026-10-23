import bench.repairs.tritonbench_g.int8_quantization.Proofs.BiasFoldIO
import VeriTile.Triton.Float.StrictBackend

namespace VeriTile.Bench.Repairs.Int8Quantization.FPEquiv
open VeriTile Triton FP.Structural Revision002
open scoped VeriTile.Spec
noncomputable section

/-- The actual candidate directly selects positive or negative 0.5 instead of multiplying 0.5
by a selected positive or negative 1. All FP32 payloads, exceptional-block handling, query/key
branches, source launch chunks, INT8/FP32 outputs and exterior frames remain.
The fixed software multiplication profile proves the only two changed local
products exactly. Other primitive instructions remain explicitly bound. -/
specification bias_fold_equiv (rest : Algebra (FP.Value .fp32))
    (ops : Model.Primitives (FP.Value .fp32)) (p : Parameters)
    (positive : 0<p.width) (block : 0<p.blockRows) :
    QuantizationIO.original (BiasFold.model rest) ops p ≡[[]]
      QuantizationIO.biasFolded (BiasFold.model rest) ops p :=
  ProgramIO.Implementation.of_equivalent (QuantizationIO.biasVariant (BiasFold.model rest) ops p)
    (QuantizationIO.bias_equivalent _ (BiasFold.model_signs rest) ops p positive block)

#print_fp_assumptions bias_fold_equiv
#axiomsClean bias_fold_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8Quantization.FPEquiv
