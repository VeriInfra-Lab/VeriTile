import bench.tritonbench_g.matrix_reduction.Kernels

/-! Shared pure source-wrapper metadata. No correctness or FP proof dependency. -/
namespace VeriTile.Bench.TritonBenchG.MatrixReduction.Host
open VeriTile.Triton TypedSource

/-- Dense strided tensor metadata for the Torch constructors used by this source. -/
structure Tensor where
  region : RegionName
  shape : List Nat
  strides : List Nat
  storage : Storage
  cuda : Bool
  device : Nat

def Tensor.stride (x : Tensor) (axisIndex : Nat) : Nat := x.strides[axisIndex]?.getD 0

structure Launch where
  grid : List Nat
  x : Tensor
  y : Tensor
  stride_xm : Nat
  stride_xn : Nat
  stride_y : Nat
  M : Nat
  N : Nat

def launchCall (x y : Tensor) (M N : Nat) : Launch :=
  ⟨[1], x, y, x.stride 0, x.stride 1, y.stride 0, M, N⟩

structure CloseOptions where
  rtol : ℚ
  atol : ℚ
  checkDType : Bool
  deriving DecidableEq

def closeOptions : CloseOptions := ⟨1/100, 1/1000, Bool.false⟩

inductive Stage where
  | dtype | randn | empty | launch | torchMax | printOptions | assertClose
  deriving DecidableEq, Repr

structure Failure where
  stage : Stage
  message : String
  deriving DecidableEq, Repr

def selectDType (name : String) : Except Failure Storage :=
  if name = "float16" then .ok .fp16
  else if name = "float32" then .ok .fp32
  else .error ⟨.dtype, name⟩

def checked (stage : Stage) (value : Except String α) : Except Failure α :=
  value.mapError fun message => ⟨stage, message⟩


end VeriTile.Bench.TritonBenchG.MatrixReduction.Host
