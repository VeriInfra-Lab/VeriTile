import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelNativeProfile
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Compiler

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.KernelCompiler
open VeriTile.Triton KernelNativeProfile KernelSource
namespace Scalar
export VeriTile.Bench.Repairs.FastLayernorm.Compiler.Shared (ScalarBinding integerBinding)
end Scalar
noncomputable section

def sourceSHA256 := "1bdf2b5ec557dcb552340ee55b376d6e8ebed924b80ef5ce40323fe84b7702ee"
def arguments (p : Parameters) : List Nat :=
  [p.heads*p.columns,p.columns,1,1,p.heads*p.columns,p.columns,1,p.heads,1,p.heads,p.columns]

inductive Storage where
  | fp32 | int32 | int8
  deriving DecidableEq
structure Signature where
  pointers : List Storage
  integers : List Scalar.ScalarBinding
  headBlock : Nat
  columnBlock : Nat
structure Options where
  target : String
  defaults : String
  numWarps : Nat
  numStages : Nat
  numCTAs : Nat
  fusion : Bool
structure PrimitiveNames where
  absolute : String
  maxNode : String
  divFull : String
  equal : String
structure Object where
  bytes : ByteArray
  sourceHash : String
  signature : Signature
  options : Options

inductive Address where
  | input | destination | output | scale
  deriving DecidableEq
def Address.offset (args : List Nat) (program destination head column : Nat) : Address → Nat
  | .input => program*args[0]?.getD 0+head*args[1]?.getD 0+column*args[2]?.getD 0
  | .destination => program*args[3]?.getD 0
  | .output => destination*args[4]?.getD 0+head*args[5]?.getD 0+column*args[6]?.getD 0
  | .scale => destination*args[7]?.getD 0+head*args[8]?.getD 0

structure Layout where
  addresses : List Address
  programAxis : Nat
  programBits : Nat
  addressBits : Nat
  destinationBits : Nat
  headSlot : Nat
  columnSlot : Nat
  reductionAxis : Nat
  sourceMasksBoth : Bool
  codeMasksBoth : Bool
  scaleMaskHeads : Bool
  zeroPadding : Bool
  zeroMaximumUsesOne : Bool
  retainsZeroScale : Bool
  fp32ScaleCasts : Bool
  /-- Selected PTX conversion is signed-16 toward zero, then a low-byte store. -/
  signed16Conversion : Bool
  lowByteStore : Bool
def layout : Layout :=
  ⟨[.input,.destination,.output,.scale],0,32,64,32,9,10,1,
    Bool.true,Bool.true,Bool.true,Bool.true,Bool.true,Bool.true,Bool.true,Bool.true,Bool.true⟩

structure ScalarInstructions where
  absolute : Word → Word
  maxNode : Word → Word → Word
  divFull : Word → Word → Word
  equal : Word → Word → Bool
structure Decoder where
  primitives : ByteArray → Signature → Options → PrimitiveNames
  layout : ByteArray → Signature → Options → Layout
  plans : ByteArray → Signature → Options → Plans
structure Platform where
  decoder : Decoder
  scalars : PrimitiveNames → ScalarInstructions
  defaults : Options
  sourceWarps : defaults.numWarps=1
  sourceStages : defaults.numStages=1

def arithmetic (platform : Platform) (object : Object) : Arithmetic :=
  let ops := platform.scalars
    (platform.decoder.primitives object.bytes object.signature object.options)
  ⟨ops.absolute,ops.maxNode,ops.divFull,ops.equal,
    platform.decoder.plans object.bytes object.signature object.options⟩

/-- Ordinary selected-object source/ABI/layout binding. The decoder sees the
object, static blocks, integer ABI and launch options, never Tensor payloads.
There is no desired result, execution success, or numerical reduction law. -/
structure Binding (platform : Platform) (p : Parameters) where
  object : Object
  source : object.sourceHash=sourceSHA256
  pointers : object.signature.pointers=[.fp32,.int32,.int8,.fp32]
  integers : List.Forall₂ Scalar.integerBinding object.signature.integers (arguments p)
  headBlock : object.signature.headBlock=p.height
  columnBlock : object.signature.columnBlock=p.width
  options : object.options=platform.defaults
  structureMatch : platform.decoder.layout object.bytes object.signature object.options=layout

theorem addresses (p : Parameters) (program destination head column : Nat) :
    layout.addresses.map (Address.offset (arguments p) program destination head column)=
      [program*(p.heads*p.columns)+head*p.columns+column,program,
       destination*(p.heads*p.columns)+head*p.columns+column,destination*p.heads+head] := by
  simp [layout,Address.offset,arguments]

theorem selected_options (platform : Platform) (p : Parameters) (binding : Binding platform p) :
    binding.object.options.numWarps=1 ∧ binding.object.options.numStages=1 := by
  rw [binding.options]
  exact ⟨platform.sourceWarps,platform.sourceStages⟩

#axiomsClean addresses
#axiomsClean selected_options
#specNonCircular Binding avoiding [kernel,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.KernelCompiler
