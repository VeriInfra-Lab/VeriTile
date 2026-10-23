/- Retained single-source native/software correctness. Original declaration
namespaces and full statements are preserved as ordinary auxiliary theorems. -/
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.NativePublicIO
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.NativeDimensionIO

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section

/-- Complete actual first-grid/second-source IO for every requested format.
FP16/BF16/FP32/FP64 codecs and all original integer/bool conversion macros
remain selected by the same Binding; every raw FP32 input word is admitted.
INT64 scratch initialization is the original logical input condition. -/
theorem global (casts : Request.Casts) (kind : Request.Kind)
    (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (g : Native.GlobalPrograms c) (positive : 0 < c.block)
    (midPositive : 0 < c.midBlock) (active : ∀ pid, pid < c.blocks → pid*c.block < c.size)
    (index64 : Bool) :
    (IO.nativeGlobal casts kind binding rest c index64).ImplementsOn
      (NativeIO.initialized casts kind binding rest c index64)
      (NativeIO.mathematical casts kind c positive midPositive index64) :=
  NativeIO.implements casts kind binding rest c g positive midPositive active index64

/-- Complete native dimension grid, including every 128-row CTA and every
loop tile, computes the independent first-NaN/first-maximum index at all output
coordinates. Primitive instruction semantics are modeling boundaries; this
source-to-native result assumes no floating rewrite or native=Real equality. -/
theorem dimension (rest : Algebra Word) (c : DimConfig) (nonempty : 0 < c.extent)
    (index64 : Bool) :
    (IO.nativeDimension rest c index64).Implements
      (NativeDimensionIO.mathematical rest c nonempty index64) :=
  NativeDimensionIO.implements rest c nonempty index64

#stmtSurfaceSubset global ⊆
  [Request.Casts, Request.Kind, BoundSource.Binding, FP.Structural.Algebra, Word, GlobalConfig,
    Native.GlobalPrograms, GlobalConfig.block, GlobalConfig.midBlock, GlobalConfig.blocks,
    GlobalConfig.size, IO.nativeGlobal, ProgramIO.ImplementsOn, NativeIO.initialized,
    NativeIO.mathematical]
#stmtSurfaceSubset dimension ⊆
  [FP.Structural.Algebra, Word, DimConfig, DimConfig.extent, IO.nativeDimension, ProgramIO.Implements,
    NativeDimensionIO.mathematical]
#specNonCircular NativeIO.mathematical avoiding [Native.global, BoundSource.Binding.first,
  BoundSource.Binding.second, FP.Structural.exec]
#specNonCircular NativeDimensionIO.mathematical avoiding [Native.dimension, TypedSource.dimension, FP.Structural.exec]
#axiomsClean global
#axiomsClean dimension
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.FPEquiv
