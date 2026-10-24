import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelNative
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelDispatch

namespace VeriTile.Bench.Repairs.LayerNormOps.KernelCompiler
open VeriTile.Triton FP.Structural KernelNative.Shared
namespace Scalar
export VeriTile.Bench.Repairs.FastLayernorm.Compiler.Shared (ScalarBinding integerBinding)
end Scalar
noncomputable section

def sourceSHA256 := "a04aa0d8020ac89ab5cd5b5798736532685a8734535e2085f7f81e942d56f795"
inductive Entry where
  | forward | backward
  deriving DecidableEq
abbrev Parameters := Sum ForwardParameters BackwardParameters
def entry : Parameters → Entry | .inl _ => .forward | .inr _ => .backward
def block : Parameters → Nat | .inl p => p.block | .inr p => p.block
def columns : Parameters → Nat | .inl p => p.columns | .inr p => p.columns
def countSlot : Parameters → Nat | .inl _ => 4 | .inr _ => 9
def epsilon : Parameters → ℝ | .inl p => p.epsilon | .inr p => p.epsilon
def pointers : Parameters → List RegionName
  | .inl p => [p.x,p.y,p.weight,p.bias,p.residual,p.residualOut,p.mean,p.rstd]
  | .inr p => [p.x,p.weight,p.bias,p.y,p.dy,p.dx,p.dw,p.db,p.dres,p.dresIn,p.mean,p.rstd]
def optionalPointer (present : Bool) : Option Nat := if present then some 32 else none
def pointerDTypes : Parameters → List (Option Nat)
  | .inl p => [some 32,some 32,some 32,optionalPointer p.hasBias,optionalPointer p.hasResidual,
      optionalPointer p.storeResidual,optionalPointer (!p.rms),some 32]
  | .inr p => [some 32,some 32,optionalPointer p.hasBias,optionalPointer p.recompute,
      some 32,some 32,some 32,optionalPointer p.hasBias,optionalPointer p.hasDres,
      optionalPointer p.storeDres,optionalPointer (!p.rms),some 32]
def arguments : Parameters → List Nat
  | .inl p => [p.xStride,p.yStride,p.residualStride,p.residualOutStride,p.columns]
  | .inr p => [p.xStride,p.yStride,p.dyStride,p.dxStride,p.dresStride,p.dresInStride,
      p.meanStride,p.rstdStride,p.rows,p.columns,p.rowsPerProgram]
def flags : Parameters → List Bool
  | .inl p => [p.rms,p.hasResidual,p.storeResidual,p.hasBias]
  | .inr p => [p.rms,p.hasDres,p.storeDres,p.hasBias,p.recompute]

theorem count_argument (p : Parameters) :
    (arguments p)[countSlot p]?=some (columns p) := by cases p <;> rfl

structure Options where
  (target defaults : String)
  (numWarps numStages numCTAs : Nat)
  fusion : Bool
structure PrimitiveNames where
  (literal add sub mul div sqrt cast fromNat fromInt neg fma epsilonABI : String)
structure Signature where
  /-- None records the original specialized None argument, not a fake pointer. -/
  pointerDTypes : List (Option Nat)
  integers : List Scalar.ScalarBinding
  epsilon : Scalar.ScalarBinding
  block : Nat
  flags : List Bool
structure Object where
  bytes : ByteArray
  sourceHash : String
  entry : Entry
  signature : Signature
  options : Options
  primitives : PrimitiveNames

inductive Address where
  | rowColumn (strideSlot : Nat)
  | column
  | row
  | rowStatistic (strideSlot : Nat)
  | programColumn (columnCountSlot : Nat)
  deriving DecidableEq
def Address.offset (arguments : List Nat) (pid row column : Nat) : Address → Nat
  | .rowColumn slot => row*(arguments[slot]?.getD 0)+column
  | .column => column
  | .row => row
  | .rowStatistic slot => row*(arguments[slot]?.getD 0)
  | .programColumn slot => pid*(arguments[slot]?.getD 0)+column
inductive CountRecipe where
  | signedI32ToFP32 (argumentSlot : Nat)
  deriving DecidableEq
structure Layout where
  addresses : List Address
  pointerBits : Nat
  offsetBits : Nat
  rowAxis : Nat
  rowGroupingSlot : Option Nat
  loopStep : Nat
  clippedRowsSlot : Option Nat
  columnMaskSlot : Nat
  /-- W's backward masked loads have an explicit zero tail. -/
  backwardWeightZero : Bool
  allStorageFP32 : Bool
  inferredArithmeticFP32 : Bool
  fp32IdentityCasts : Bool
  countConversion : CountRecipe

def layout : Parameters → Layout
  | .inl _ => ⟨[.rowColumn 0,.rowColumn 1,.column,.column,
      .rowColumn 2,.rowColumn 3,.row,.row],64,32,0,none,1,none,4,Bool.false,Bool.true,Bool.true,Bool.true,
      .signedI32ToFP32 4⟩
  | .inr _ => ⟨[.rowColumn 0,.column,.column,.rowColumn 1,
      .rowColumn 2,.rowColumn 3,.programColumn 9,.programColumn 9,
      .rowColumn 4,.rowColumn 5,.rowStatistic 6,.rowStatistic 7],
      64,32,0,some 10,1,some 8,9,Bool.true,Bool.true,Bool.true,Bool.true,
      .signedI32ToFP32 9⟩

theorem forward_address_offsets (p : ForwardParameters) (row column : Nat) :
    (layout (.inl p)).addresses.map (Address.offset (arguments (.inl p)) row row column)=
      [row*p.xStride+column,row*p.yStride+column,column,column,
       row*p.residualStride+column,row*p.residualOutStride+column,row,row] := rfl

theorem backward_address_offsets (p : BackwardParameters) (pid row column : Nat) :
    (layout (.inr p)).addresses.map (Address.offset (arguments (.inr p)) pid row column)=
      [row*p.xStride+column,column,column,row*p.yStride+column,
       row*p.dyStride+column,row*p.dxStride+column,pid*p.columns+column,pid*p.columns+column,
       row*p.dresStride+column,row*p.dresInStride+column,row*p.meanStride,row*p.rstdStride] := rfl

/-- Object inspection returns structure only. Reductions inspect Shape with
numeric leaf payloads erased; Schedule preserves selected operand order and
FMA/subtraction choices. No decoder receives the Tensor values. -/
structure Decoder where
  layout : ByteArray → Signature → Options → Layout
  reductions : ByteArray → Signature → Options → PrimitiveNames → Reductions
  schedule : ByteArray → Signature → Options → PrimitiveNames → OutputSchedule

structure Platform where
  decoder : Decoder
  primitives : PrimitiveNames → Instructions
  hostReal : UInt64 → Option ℝ
  epsilon : Object → UInt64 → Word
  epsilonBinding : Scalar.ScalarBinding → UInt64 → Prop
  defaults : Entry → Nat → Nat → Options
  sourceWarps : ∀ e b w,(defaults e b w).numWarps=w

/-- Selected source/ABI/options and structural object binding. Integer and
pointer slots follow the original Python call order; epsilon remains the
actual builtin binary64 argument, with a separate scalar conversion law. -/
structure Binding (platform : Platform) (p : Parameters) (host : UInt64)
    (R : Reductions) (S : OutputSchedule) where
  object : Object
  source : object.sourceHash=sourceSHA256
  entryMatch : object.entry=entry p
  blockMatch : object.signature.block=block p
  pointerTypes : object.signature.pointerDTypes=pointerDTypes p
  integers : List.Forall₂ Scalar.integerBinding object.signature.integers (arguments p)
  countRuntime : object.signature.integers[countSlot p]?=some (.runtime .i32)
  flagsMatch : object.signature.flags=flags p
  epsilonABI : platform.epsilonBinding object.signature.epsilon host
  hostValue : platform.hostReal host=some (epsilon p)
  sourceConfiguration : object.options.numWarps∈[1,2,4,8,16,32]
  options : object.options=platform.defaults (entry p) (block p) object.options.numWarps
  addressing : platform.decoder.layout object.bytes object.signature object.options=layout p
  trees : platform.decoder.reductions object.bytes object.signature object.options object.primitives=R
  order : platform.decoder.schedule object.bytes object.signature object.options object.primitives=
    fun r i => KernelNative.runtimeSchedule (S r i)

def CountConversion (platform : Platform) (p : Parameters) (host : UInt64)
    (binding : Binding platform p host R S) : Prop :=
  KernelNative.CountConversion (platform.primitives binding.object.primitives) (columns p)

/-- Separate scalar-only laws: actual host epsilon conversion and actual
runtime signed-i32 count conversion. Neither follows merely from a range. -/
structure Conversion (platform : Platform) (p : Parameters) (host : UInt64)
    (binding : Binding platform p host R S) : Prop where
  epsilon : literal32 (platform.primitives binding.object.primitives) (epsilon p)=platform.epsilon binding.object host
  count : CountConversion platform p host binding

def forward (platform : Platform) (p : ForwardParameters) (host : UInt64)
    (R : Reductions) (S : OutputSchedule) (binding : Binding platform (.inl p) host R S)
    (conversion : Conversion platform (.inl p) host binding)
    (s : State Term) (h : KernelExecution.ForwardAdmissible p) :=
  KernelNative.forward R (platform.primitives binding.object.primitives) S p s h
    conversion.count (platform.epsilon binding.object host) conversion.epsilon

def backward (platform : Platform) (p : BackwardParameters) (host : UInt64)
    (R : Reductions) (S : OutputSchedule) (binding : Binding platform (.inr p) host R S)
    (conversion : CountConversion platform (.inr p) host binding)
    (s : State Term) (h : KernelReadback.Admissible p) :=
  KernelNative.backward R (platform.primitives binding.object.primitives) S p s h conversion

#axiomsClean forward
#axiomsClean backward
#axiomsClean forward_address_offsets
#axiomsClean backward_address_offsets
#axiomsClean count_argument
end
end VeriTile.Bench.Repairs.LayerNormOps.KernelCompiler
