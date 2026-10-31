import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_SourceGridExecution
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_StructuralControl
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_HostPlan

/-! Shared source definitions re-exported by Kernels.lean. The definitions are
moved verbatim from the earlier tile/host helper modules; their public names and
bodies are unchanged. Both proof entries execute this one FP32-typed source.
The signed block-pointer lowering retains both full signed address sums, logical
boundary masks, unspecified padding, both pointer-element casts and every source
control statement. The host keeps exact partition/guard/argument/return order. -/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open VeriTile.Triton FP.Structural SignedView SignedGrid HostPlan SourceSignedBranch
open StructuralGrid StructuralControl
namespace Whole
/-- Selected-FP32 constructor metadata. The original base object is retained
for unwrap; tensor is precisely the metadata consumed by the rank-one wrapper. -/
structure WrappedBuffer where
  baseTensor : Buffer
  tensor : Buffer

def wrapBuffer (base : Buffer) (shape : Option (List Nat))
    (strides : Option (List Int)) (offset : Int) : WrappedBuffer :=
  ⟨base, reinterpret base shape strides offset⟩

def WrappedBuffer.dtype (_x : WrappedBuffer) : FloatDType := .fp32
def WrappedBuffer.stride (x : WrappedBuffer) : List Int := x.tensor.strides
def WrappedBuffer.size (x : WrappedBuffer) : List Nat := x.tensor.shape
def WrappedBuffer.elementSize (_x : WrappedBuffer) : Nat := 4
def WrappedBuffer.numel (x : WrappedBuffer) : Nat := x.tensor.shape.prod
def WrappedBuffer.dim (x : WrappedBuffer) : Nat := x.tensor.shape.length
def WrappedBuffer.unwrap (x : WrappedBuffer) : Buffer := x.baseTensor
/-- The host computes the signed byte offset as a Python integer before native
argument admission; NativeAccess proves the finite address of valid views. -/
def WrappedBuffer.dataPtr (x : WrappedBuffer) (allocationBase : Nat) : Int :=
  (allocationBase : Int) + 4 * x.tensor.base
end Whole

noncomputable section
namespace StructuralTile

end StructuralTile
namespace StructuralHost
open StructuralTile
def views (a : Arguments) : SignedViews := ⟨a.input, a.output, a.n, a.B⟩

def idsPerProgram (a : Arguments) (pid : Nat) : List Nat :=
  if a.oneTile then [pid] else (List.range a.T).map (fun j => pid + j * a.C)

def sourceTileIds (a : Arguments) : List Nat :=
  (List.range a.C).flatMap (idsPerProgram a)

inductive Result (α : Type) where
  | hostFailure (failure : Failure) (state : State α)
  | executionFailure
  | returned (output : Buffer) (state : State α)

end StructuralHost
end
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer
