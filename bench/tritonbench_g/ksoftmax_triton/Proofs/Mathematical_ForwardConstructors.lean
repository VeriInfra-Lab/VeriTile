import bench.tritonbench_g.ksoftmax_triton.Proofs.Mathematical_ForwardSource

/-! Storage-profile labels retained for the existing typed model. Concrete
forward execution uses the shared selected source through `fp32Original`. -/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Typed
open VeriTile.Triton

inductive CheckedProfile where
  | fp32 | narrow | mixed | wide
  deriving DecidableEq, Repr

end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Typed
