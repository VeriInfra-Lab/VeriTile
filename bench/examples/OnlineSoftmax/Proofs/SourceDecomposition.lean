import bench.examples.OnlineSoftmax.Kernels

/- Proof-only prefixes of the complete source. The authoritative programs
remain the two raw Python bodies in Kernels; full_split connects every prefix
and suffix used by the execution proofs to that actual complete program. -/
namespace VeriTile.Bench.Examples.OnlineSoftmax.SourceDecomposition
open VeriTile.Triton
open VeriTile.Bench.Examples.OnlineSoftmax.Kernels

def normalizerSource (x y : RegionName) (N : Nat) : ComputeKernel :=
  .mk [x] [] ((onlineSoftmaxKernel x y N).surfaceBody.take 4)

def loadedNormalizerSource (x y : RegionName) (N : Nat) : ComputeKernel :=
  .mk [x] [] ((onlineSoftmaxKernel x y N).surfaceBody.take 7)

def normalizationCode (x y : RegionName) (N : Nat) : List ComputeStmt :=
  (onlineSoftmaxKernel x y N).surfaceBody.drop 4

def rowLoadCode (x y : RegionName) (N : Nat) : List ComputeStmt :=
  ((onlineSoftmaxKernel x y N).surfaceBody.drop 4).take 3

theorem loaded_prefix (x y : RegionName) (N : Nat) :
    (loadedNormalizerSource x y N).surfaceBody =
      (normalizerSource x y N).surfaceBody ++ rowLoadCode x y N := rfl

theorem full_split (x y : RegionName) (N : Nat) :
    onlineSoftmaxKernel x y N = .mk [x] [y]
      ((normalizerSource x y N).surfaceBody ++ normalizationCode x y N) := rfl

theorem loaded_split (x y : RegionName) (N : Nat) :
    onlineSoftmaxKernel x y N = .mk [x] [y]
      ((loadedNormalizerSource x y N).surfaceBody ++
        (onlineSoftmaxKernel x y N).surfaceBody.drop 7) := rfl

end VeriTile.Bench.Examples.OnlineSoftmax.SourceDecomposition
