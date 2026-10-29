import bench.tritonbench_g.rotary_transform_ops.RotaryTransformOps
import bench.tritonbench_g.rotary_transform.Proofs.FPSoftware
import bench.tritonbench_g.rotary_transform.Proofs.FPShape

/-! The two Python kernels have identical selected non-interleaved paired-row
and metadata-row source statements. Reuse their typed execution and independent
rounded specification, and establish this task's own complete source bridge. -/
namespace VeriTile.Bench.TritonBenchG.RotaryTransformOps.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
abbrev Kind := RotaryTransform.FPRow.Kind
abbrev Parameters := RotaryTransform.FPRow.Parameters

def source (k : Kind) (p : Parameters) : ComputeKernel := RotaryTransform.FPRow.source k p

def legacySource (k : Kind) (p : Parameters) : ComputeKernel :=
  match k with
  | .row => rotary_kernel_o0o1_row p.output p.input p.cosine p.sine
      p.sequenceOffset p.sequenceLength p.halfDimension p.cosineLength
      p.strideOutBatch p.strideOutSequence p.strideOutHead p.strideOutDimension
      p.strideInBatch p.strideInSequence p.strideInHead p.strideInDimension p.blockRows p.width
  | .fixed | .varlen => rotary_meta_row p.output p.input p.cosine p.sine p.cuLengths
      p.headIndex p.sequenceOffset p.sequenceLength p.halfDimension p.cosineLength
      p.strideOutBatch p.strideOutSequence p.strideOutHead p.strideOutDimension
      p.strideInBatch p.strideInSequence p.strideInHead p.strideInDimension p.blockRows p.width (k == .varlen)

theorem source_projection (k : Kind) (p : Parameters) :
    (source k p).eraseDType.surfaceBody = (legacySource k p).eraseDType.surfaceBody := by
  calc
    _ = (RotaryTransform.FPRow.legacySource k p).eraseDType.surfaceBody :=
      RotaryTransform.FPRow.source_projection k p
    _ = _ := by cases k <;> rfl

theorem software_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (k : Kind) (p : Parameters)
    (s : State (FP.Value .fp32)) (hstride : p.strideOutDimension ≠ 0) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (source k p) s = some final ∧
      RotaryTransform.FPRow.Post k p s final (RotaryTransform.FPRow.roundedValue cfg k p s) :=
  RotaryTransform.FPRow.software_correctness rest cfg k p s hstride

#axiomsClean source_projection
#axiomsClean software_correctness
end
end VeriTile.Bench.TritonBenchG.RotaryTransformOps.FPStages
