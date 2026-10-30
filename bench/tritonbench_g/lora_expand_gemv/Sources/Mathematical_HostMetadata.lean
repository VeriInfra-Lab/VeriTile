import bench.tritonbench_g.lora_expand_gemv.Sources.Mathematical_BranchSource

/-! Ordered logical metadata semantics of the original `_bgmv_expand` wrapper.
Tensor shapes/strides are mathematical metadata; region names already denote
tensor data pointers. Native tensor/byte-memory/ABI binding remains separate.
Unknown output and index dtypes are retained, not falsely rejected by an
assertion absent from the Python source. -/
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Host
open VeriTile.Triton
open FPValue (OutputFormat)
open Branches (InputProfile Parameters)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

inductive DType where
  | half | bfloat | single | other (tag : String)
  deriving DecidableEq, Repr

def outputDType : OutputFormat → DType
  | .half => .half | .bfloat => .bfloat | .single => .single

structure Tensor where
  region : RegionName
  dtype : DType
  /-- Each dimension stores its size and element stride together. -/
  dimensions : List (Nat × Nat)
  deriving DecidableEq, Repr

def Tensor.rank (t : Tensor) := t.dimensions.length
def Tensor.size? (t : Tensor) (i : Nat) := (t.dimensions[i]?).map Prod.fst
def Tensor.stride? (t : Tensor) (i : Nat) := (t.dimensions[i]?).map Prod.snd
def Tensor.lastSize? (t : Tensor) := t.dimensions.getLast?.map Prod.fst

/-- Standard contiguous layout ignores size-one dimensions. Empty tensors
are contiguous independently of their strides. -/
def contiguousAux : List (Nat × Nat) → Nat → Bool
  | [], _ => Bool.true
  | (size,stride) :: dims, expected =>
      if size = 1 then contiguousAux dims expected
      else decide (stride = expected) && contiguousAux dims (expected * size)

def Tensor.contiguous (t : Tensor) : Bool :=
  t.dimensions.any (fun dim => decide (dim.1 = 0)) || contiguousAux t.dimensions.reverse 1

def Tensor.squeezeOne (t : Tensor) : Tensor :=
  { t with dimensions := t.dimensions.eraseIdx 1 }

structure Arguments where
  (inputs weights output indices : Tensor)
  addInputs : Bool := Bool.true

inductive InputDType where
  | half | bfloat | single
  deriving DecidableEq, Repr
inductive WeightDType where
  | half | bfloat
  deriving DecidableEq, Repr

def profile : InputDType → WeightDType → InputProfile
  | .half, .half => .halfHalf | .half, .bfloat => .halfBfloat
  | .bfloat, .half => .bfloatHalf | .bfloat, .bfloat => .bfloatBfloat
  | .single, .half => .singleHalf | .single, .bfloat => .singleBfloat

inductive HostError where
  | inputDTypeAssert | weightDTypeAssert
  | inputSizeOne | weightLastSize | innerSizeAssert
  | inputContiguousAssert | outputContiguousAssert
  | weightSingletonAssert | weightRankAssert | weightContiguousAssert
  | weightShapeUnpack | evenKZeroDivision | indexSizeZero
  | inputStrideZero | inputStrideOne
  | weightStrideZero | weightStrideOne | weightStrideTwo
  | outputStrideZero | outputStrideOne
  deriving DecidableEq, Repr

def checkInput : DType → Except HostError InputDType
  | .half => .ok .half | .bfloat => .ok .bfloat | .single => .ok .single
  | .other _ => .error .inputDTypeAssert
def checkWeight : DType → Except HostError WeightDType
  | .half => .ok .half | .bfloat => .ok .bfloat
  | _ => .error .weightDTypeAssert

def require (b : Bool) (error : HostError) : Except HostError Unit :=
  if b then .ok () else .error error
def get (value : Option Nat) (error : HostError) : Except HostError Nat :=
  match value with | some n => .ok n | none => .error error

/-- Exact six-shift helper used by the installed Triton Python implementation.
Its zero input returns zero. We deliberately do not replace the fixed six
shifts by an ideal unbounded ceil-power-of-two function. -/
def nextPowerOfTwo (n : Nat) : Nat :=
  if n = 0 then 0 else
    let x := n - 1
    let x := x ||| (x >>> 1)
    let x := x ||| (x >>> 2)
    let x := x ||| (x >>> 4)
    let x := x ||| (x >>> 8)
    let x := x ||| (x >>> 16)
    let x := x ||| (x >>> 32)
    x + 1

def chooseCast : InputDType → Bool
  | .single => Bool.true | _ => Bool.false

structure Launch where
  inputType : InputDType
  weightType : WeightDType
  weightView : Tensor
  (outputType indexType : DType)
  parameters : Parameters
  batches : Nat

def Launch.grid (l : Launch) : Nat × Nat := (l.parameters.outputSize, l.batches)
def Launch.profile (l : Launch) : InputProfile := Host.profile l.inputType l.weightType

/-- The actual Python order: dtype assertions; size accesses/equality;
contiguity assertions; 4D singleton squeeze or 3D assertion; weight
contiguity; N/K and next-power metadata; EVEN_K (which can raise); index
batch count; and finally left-to-right call-argument stride accesses. -/
def prepare (a : Arguments) : Except HostError Launch := do
  let inputType ← checkInput a.inputs.dtype
  let weightType ← checkWeight a.weights.dtype
  let inputK ← get (a.inputs.size? 1) .inputSizeOne
  let weightK ← get a.weights.lastSize? .weightLastSize
  require (decide (inputK = weightK)) .innerSizeAssert
  require a.inputs.contiguous .inputContiguousAssert
  require a.output.contiguous .outputContiguousAssert
  let weights ← if a.weights.rank = 4 then do
      let singleton ← get (a.weights.size? 1) .weightSingletonAssert
      require (decide (singleton = 1)) .weightSingletonAssert
      pure a.weights.squeezeOne
    else do
      require (decide (a.weights.rank = 3)) .weightRankAssert
      pure a.weights
  require weights.contiguous .weightContiguousAssert
  let (n,k) ← match weights.dimensions.reverse with
    | (k,_) :: (n,_) :: _ => pure (n,k)
    | _ => .error .weightShapeUnpack
  let blockK := nextPowerOfTwo k
  if blockK = 0 then .error .evenKZeroDivision else do
    let evenK := decide (k % blockK = 0)
    let batches ← get (a.indices.size? 0) .indexSizeZero
    let xm ← get (a.inputs.stride? 0) .inputStrideZero
    let xk ← get (a.inputs.stride? 1) .inputStrideOne
    let l0 ← get (weights.stride? 0) .weightStrideZero
    let lk ← get (weights.stride? 1) .weightStrideOne
    let ln ← get (weights.stride? 2) .weightStrideTwo
    let cm ← get (a.output.stride? 0) .outputStrideZero
    let cn ← get (a.output.stride? 1) .outputStrideOne
    pure {
      inputType := inputType, weightType := weightType, weightView := weights,
      outputType := a.output.dtype, indexType := a.indices.dtype,
      batches := batches,
      parameters := {
        input := a.inputs.region, weight := weights.region, output := a.output.region,
        indices := a.indices.region.cast, inner := k, outputSize := n, splits := n,
        inputBatchStride := xm, inputInnerStride := xk, loraStride := l0,
        weightRowStride := lk, weightInnerStride := ln,
        outputBatchStride := cm, outputStride := cn,
        blockRows := 256, blockInner := blockK, evenK := evenK,
        addInputs := a.addInputs, castType := chooseCast inputType } }

theorem nextPowerOfTwo_zero : nextPowerOfTwo 0 = 0 := rfl
theorem nextPowerOfTwo_pos (n : Nat) (hn : 0 < n) : 0 < nextPowerOfTwo n := by
  simp only [nextPowerOfTwo, if_neg (by omega : n ≠ 0)]
  omega

theorem squeeze_four_contiguous (r : RegionName) (d : DType)
    (l sl oneStride n sn k sk : Nat) :
    (Tensor.squeezeOne ⟨r,d,[(l,sl),(1,oneStride),(n,sn),(k,sk)]⟩).contiguous =
      (Tensor.mk r d [(l,sl),(n,sn),(k,sk)]).contiguous := rfl

#axiomsClean nextPowerOfTwo_zero
#axiomsClean nextPowerOfTwo_pos
#axiomsClean squeeze_four_contiguous
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Host
