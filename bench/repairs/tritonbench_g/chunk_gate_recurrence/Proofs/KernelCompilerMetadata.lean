import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelNativeSpec

/-! Selected-object boundary. Decoder inputs are object bytes, ABI metadata
and compiler options; tensor payloads and output assertions are absent.
Scalar instructions and their FP32 bit semantics are platform model data. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelCompiler
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelCarrier
noncomputable section

inductive Integer where
  | runtimeI32 (divisibility : Nat)
  | specialized (value : Nat)
  deriving DecidableEq
def admits : Integer → Nat → Prop
  | .runtimeI32 divisor,n => n<2^31 ∧ 0<divisor ∧ n%divisor=0
  | .specialized value,n => value=n
inductive Parameter where
  | pointer (role : RegionName)
  | integer (name : String) (binding : Integer)
  | constexpr (name : String) (value : Nat)
  | absentSeed
  deriving DecidableEq
def parameters (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (H N : Integer) : List Parameter :=
  let tail := [Parameter.integer "NUM_HEAD" H,.integer "NUM_BLOCK" N,
    .constexpr "D_MODEL_K" c.K,.constexpr "D_MODEL_V" c.V,
    .constexpr "BLOCK_MODEL_K" 64,.constexpr "BLOCK_MODEL_V" 16]
  match kind with
  | .forward => [.pointer "S",.pointer "D",.pointer "O"]++tail++
    [if seed then .pointer "Seed" else .absentSeed,.constexpr "SEED_B" s.batch,
      .constexpr "SEED_H" s.head,.constexpr "SEED_K" s.key,.constexpr "SEED_V" s.value]
  | .backward => [.pointer "S",.pointer "D",.pointer "DI",.pointer "DG",.pointer "DL",.pointer "DS"]++tail
structure Signature where
  parameters : List Parameter
  pointerBits : Nat
  elementBits : Nat
  pointerDivisibility : RegionName → Nat
structure Options where
  backend : String
  architecture : Nat
  compiler : String
  numWarps : Nat
  numStages : Nat
  numCTAs : Nat
  fusion : Bool
structure PrimitiveNames where
  add : String
  multiply : String
  fusedMultiplyAdd : String
structure Object where
  bytes : ByteArray
  sourceHash : String
  entry : Kind
  signature : Signature
  options : Options
  primitives : PrimitiveNames
structure Layout where
  kind : Kind
  K : Nat
  V : Nat
  BK : Nat
  BV : Nat
  seed : Bool
  strides : SeedStrides
  indexBits : Nat
  pointerBits : Nat
def layout (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) : Layout :=
  match kind with
  | .forward => ⟨kind,c.K,c.V,64,16,seed,s,32,64⟩
  | .backward => ⟨kind,c.K,c.V,64,16,Bool.false,⟨0,0,0,0⟩,32,64⟩
def template (recipe : Layout) (H N : Nat) : ComputeKernel := match recipe.kind with
  | .forward => Kernels.forward "S" "D" "O" "Seed" H N recipe.K recipe.V
      recipe.strides.batch recipe.strides.head recipe.strides.key recipe.strides.value recipe.seed
  | .backward => Kernels.backward "S" "D" "DI" "DG" "DL" "DS" H N recipe.K recipe.V recipe.BK recipe.BV
theorem template_source (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) :
    template (layout kind c s seed) c.H c.N=source kind c s seed := by cases kind <;> rfl

structure Decoder where
  layout : ByteArray → Signature → Options → Layout
  swapped : ByteArray → Signature → Options → PrimitiveNames → Bool
  reduction : ByteArray → Signature → Options → PrimitiveNames → Reduction.ProductTree 1024
structure Platform where
  decoder : Decoder
  instructions : PrimitiveNames → Instructions
  defaults : Kind → Options

structure Binding (platform : Platform) (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) where
  object : Object
  H : Integer
  N : Integer
  source : object.sourceHash=KernelProgram.sourceSHA256
  entry : object.entry=kind
  parameters : object.signature.parameters=KernelCompiler.parameters kind c s seed H N
  headArgument : admits H c.H
  timeArgument : admits N c.N
  headOne : c.H=1 → H=.specialized 1
  timeOne : c.N=1 → N=.specialized 1
  pointerBits : object.signature.pointerBits=64
  elementBits : object.signature.elementBits=32
  options : object.options=platform.defaults kind
  backend : object.options.backend="cuda"
  compiler : object.options.compiler="3.7.1"
  warps : object.options.numWarps=4
  stages : object.options.numStages=3
  ctas : object.options.numCTAs=1
  fusion : object.options.fusion=Bool.true
  decoded : platform.decoder.layout object.bytes object.signature object.options=layout kind c s seed
  operands : platform.decoder.swapped object.bytes object.signature object.options object.primitives=swapped kind c
  coverage : (platform.decoder.reduction object.bytes object.signature object.options object.primitives).leaves.Perm
    (List.finRange 1024)

def plan (platform : Platform) (binding : Binding platform kind c s seed) : Reduction.ProductPlan 1024 :=
  ⟨platform.decoder.reduction binding.object.bytes binding.object.signature binding.object.options binding.object.primitives,
    binding.coverage⟩
def algebra (platform : Platform) (binding : Binding platform kind c s seed) : Algebra Node :=
  KernelCarrier.algebra (platform.instructions binding.object.primitives) (plan platform binding)
    (platform.decoder.swapped binding.object.bytes binding.object.signature binding.object.options binding.object.primitives)
def decodedSource (platform : Platform) (binding : Binding platform kind c s seed) : ComputeKernel :=
  template (platform.decoder.layout binding.object.bytes binding.object.signature binding.object.options) c.H c.N
theorem decoded_source (platform : Platform) (binding : Binding platform kind c s seed) :
    decodedSource platform binding=source kind c s seed := by
  unfold decodedSource
  rw [binding.decoded,template_source]

structure Facts (platform : Platform) (binding : Binding platform kind c s seed)
    (p : Program c) (u v : State Node) : Prop extends
    KernelExecution.Facts (algebra platform binding) kind c s seed p u v where
  decodedSuccess : FP.Structural.exec (engine (algebra platform binding)) (decodedSource platform binding) u=some v

#axiomsClean decoded_source
#specNonCircular Binding avoiding [FP.Structural.exec,FP.Structural.run,KernelNative.expected]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelCompiler
