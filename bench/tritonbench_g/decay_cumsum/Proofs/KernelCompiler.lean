import bench.tritonbench_g.decay_cumsum.Proofs.KernelNative
import bench.tritonbench_g.decay_cumsum.Proofs.KernelContract

/-! A selected object is consumed by the actual source interpreter. Decoder
ports inspect bytes, ABI and options only; they never receive input payloads.
Instruction/conversion interpretations are model data, not output axioms. -/
namespace VeriTile.Bench.Repairs.DecayCumsum.KernelCompiler
open VeriTile.Triton FP.Structural FP.ProfiledRewrite Geometry Native
noncomputable section

def sourceSHA256 := "8280a38504ba5e0d5fb398180cc30cb90230739a6ca545e2c455e1d6cc309974"
inductive ScalarType where
  | i1 | i32 | i64 | u32 | u64 | fp32
  deriving DecidableEq
inductive ScalarBinding where
  | runtime (dtype : ScalarType)
  | specialized (value : Int)
  deriving DecidableEq
def integerBinding : ScalarBinding → Nat → Prop
  | .runtime .i1,n => n≤1
  | .runtime .i32,n => n<2^31
  | .runtime .i64,n => n<2^63
  | .runtime .u32,n => n<2^32
  | .runtime .u64,n => n<2^64
  | .specialized value,n => value=(n:Int)
  | _,_ => False
def runtimeT (h : Call) (p : Phase) := if p.span=h.BT then h.T else p.span
def arguments (h : Call) (p : Phase) : List Nat :=
  [h.headStride,h.width,1,h.B,h.H,runtimeT h p]
def scaleArgument (kind : Kind) : Nat := (argumentRoles kind).length+6
theorem scale_argument_slots : scaleArgument .forward=8 ∧ scaleArgument .prepare=11 ∧ scaleArgument .backward=14 := by
  exact ⟨rfl,rfl,rfl⟩
structure Signature where
  pointers : List RegionName
  pointerBits : Nat
  elementBits : Nat
  pointerDivisibility : List Nat
  integers : List ScalarBinding
  integerDivisibility : List Nat
  scale : ScalarBinding
  BT : Nat
  BK : Nat
  DK : Nat
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
  sub : String
  mul : String
  fma : String
  neg : String
  exp2 : String
  scaleABI : String
structure Object where
  bytes : ByteArray
  sourceHash : String
  entry : Kind
  signature : Signature
  options : Options
  primitives : PrimitiveNames
structure Layout where
  kind : Kind
  timeSpan : Nat
  lanes : Nat
  width : Nat
  headStrideSlot : Nat
  programIdsSigned32 : Bool
  programIdsWidened64 : Bool
  loopIndices64 : Bool
  offsetBits : Nat
  fp32Memory : Bool
  runtimeScaleSlot : Bool
  scaleArgument : Nat
  /-- The scalar syntax marker denotes the runtime argument, never constant 2. -/
  scaleMarker : ℝ
def layout (kind : Kind) (h : Call) (p : Phase) : Layout :=
  ⟨kind,p.span,h.BK,h.width,0,Bool.true,Bool.true,Bool.true,64,Bool.true,Bool.true,scaleArgument kind,scaleSlot⟩

structure Decoder where
  layout : ByteArray → Signature → Options → Layout
  sites : ByteArray → Signature → Options → PrimitiveNames → Graph
structure Platform where
  decoder : Decoder
  instructions : PrimitiveNames → Instructions
  /-- Actual Python binary64 argument packed into the selected FP32 ABI.
  This local conversion function includes infinities, NaNs and underflow. -/
  scale : Object → UInt64 → Word
  defaults : Kind → Options

structure Binding (platform : Platform) (kind : Kind) (h : Call) (p : Phase) where
  object : Object
  source : object.sourceHash=sourceSHA256
  entry : object.entry=kind
  pointers : object.signature.pointers=argumentRoles kind
  pointerBits : object.signature.pointerBits=64
  storage : object.signature.elementBits=32
  integers : List.Forall₂ integerBinding object.signature.integers (arguments h p)
  integerAttributes : List.Forall₂ (fun divisor value => 0<divisor ∧ value%divisor=0)
    object.signature.integerDivisibility (arguments h p)
  scale : object.signature.scale=.runtime .fp32
  time : object.signature.BT=p.span
  lanes : object.signature.BK=h.BK
  width : object.signature.DK=h.width
  options : object.options=platform.defaults kind
  backend : object.options.backend="cuda"
  version : object.options.compiler="3.7.1"
  warps : object.options.numWarps=4
  stages : object.options.numStages=3
  ctas : object.options.numCTAs=1
  fusion : object.options.fusion=Bool.true
  decoded : platform.decoder.layout object.bytes object.signature object.options=layout kind h p
  graphs : ∀ site,platform.decoder.sites object.bytes object.signature object.options object.primitives site=selected site

def algebra (platform : Platform) (binding : Binding platform kind h p) (host : UInt64) : Algebra Node :=
  Native.algebra (platform.instructions binding.object.primitives)
    (platform.decoder.sites binding.object.bytes binding.object.signature binding.object.options binding.object.primitives)
    kind (platform.scale binding.object host)

theorem runtime_scale_value (platform : Platform) (binding : Binding platform kind h p) (host : UInt64) :
    ((algebra platform binding host).literal (some .fp32) .real scaleSlot).value=platform.scale binding.object host :=
  scale_all_words _

#axiomsClean runtime_scale_value
#axiomsClean scale_argument_slots
end
end VeriTile.Bench.Repairs.DecayCumsum.KernelCompiler
