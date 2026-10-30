import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.Native
namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.SourceNative
open VeriTile.Triton FP.Structural StaticChecked.Native
noncomputable section
/-- The raw body explicitly converts the dot result to FP16 before storing.
The existing native model already implements FP32-to-FP16 and FP32 identity.
Its FP16 storage identity is defined here as an identity on the encoded word;
no arbitrary backend conversion is assumed to have this property. -/
def algebra (rest : Algebra Word) (I : Instructions) : Algebra Word :=
  { StaticChecked.Native.algebra rest I with cast := fun p src dst value =>
      match src,dst with
      | .fp16,.fp16 => value
      | src,dst => (StaticChecked.Native.algebra rest I).cast p src dst value }
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.SourceNative
