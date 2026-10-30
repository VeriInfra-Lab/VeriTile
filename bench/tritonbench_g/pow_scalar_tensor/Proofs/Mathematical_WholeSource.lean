import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_StructuralControl
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_SourceSignedBranch
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_HostPlan

/-! Selected FP32 source for scalar-base power. Signed block pointers are
lowered to the COMPLETE signed address and the unchanged logical boundary mask.
The `.real` register channel carries FP32 compute, explicitly marked `.fp32`;
`.fp32` memory channels retain storage. The two element casts and scalar cast
are the three casts in the Python source, with no inserted arithmetic.
The shared host metadata is the identical generated rank-one wrapper. -/
namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.WholeSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open HostPlan SignedView SourceSignedBranch StructuralControl

structure Config where
  inputView : View
  outputView : View
  n : Nat
  B : Nat
  base : ℝ
  candidate : Bool

def offset (c : Config) : Op .nat [] :=
  if c.candidate then .mul .nat .nil (.constNat c.B) (.ref .nat [] "tile_id0")
  else .mul .nat .nil (.ref .nat [] "tile_id0") (.constNat c.B)

def Config.perTile (c : Config) : List ComputeStmt :=
  [.alg (.assign .nat [] "tile_id0" (.ref .nat [] "tile_id")),
   .alg (.assign .nat [] "offset0" (offset c)),
   .alg (.assign .nat [c.B] "in0_bptr" (dynamicAddress c.inputView c.B)),
   .assign .real [c.B] "in0" (.compute (.alg .fp32 (.castFloat .fp32 .real
     (.load .fp32 (.region c.inputView.region (.ref .nat [c.B] "in0_bptr"))
       (.mask (dynamicMask c.n c.B)) .unspecified)))),
   .assign .real [c.B] "out0" (.compute (.alg .fp32 (.pow .scalarL
     (.castFloat .real .real (.const c.base)) (.ref .real [c.B] "in0")))),
   .alg (.assign .nat [c.B] "out0_bptr" (dynamicAddress c.outputView c.B)),
   .store .fp32 [c.B] (.region c.outputView.region (.ref .nat [c.B] "out0_bptr"))
     (.alg (.castFloat .real .fp32 (.withPrecision .real .fp32 (.ref .real [c.B] "out0"))))
     (.mask (dynamicMask c.n c.B))]

def kernel (c : Config) (oneTile : Bool) (tiles : Nat) : ComputeKernel :=
  .mk [c.inputView.region] [c.outputView.region]
    (if oneTile then oneTileBody c.perTile c.n c.B else gridBody c.perTile c.n c.B tiles)

def config (base : ℝ) (candidate : Bool) (a : Arguments) : Config :=
  ⟨a.input, a.output, a.n, a.B, base, candidate⟩

def idsPerProgram (a : Arguments) (pid : Nat) : List Nat :=
  if a.oneTile then [pid] else (List.range a.T).map (fun j => pid + j * a.C)

def sourceTileIds (a : Arguments) : List Nat :=
  (List.range a.C).flatMap (idsPerProgram a)

noncomputable def executeArguments {α : Type} [Inhabited α] (M : Algebra α)
    (base : ℝ) (candidate : Bool) (a : Arguments) (s : State α) : Option (State α) :=
  runPrograms M (kernel (config base candidate a) a.oneTile a.T).surfaceBody
    a.C (List.range a.C) s

inductive Result (α : Type) where
  | hostFailure (failure : Failure) (state : State α)
  | executionFailure
  | returned (output : Buffer) (state : State α)

noncomputable def runHost {α : Type} [Inhabited α] (M : Algebra α)
    (base : ℝ) (candidate : Bool) (input output : Buffer)
    (guardSucceeds : Bool) (s : State α) : Result α :=
  match prepare input output guardSucceeds with
  | .error failure => .hostFailure failure s
  | .ok a => match executeArguments M base candidate a s with
    | none => .executionFailure
    | some final => .returned output final

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
def WrappedBuffer.dataPtr (x : WrappedBuffer) (allocationBase : Nat) : Int :=
  (allocationBase : Int) + 4 * x.tensor.base

end VeriTile.Bench.TritonBenchG.PowScalarTensor.WholeSource
