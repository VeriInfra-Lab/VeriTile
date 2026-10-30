import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelNative
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelContract
/-! Selected-object interpretation. Decoder ports take only object bytes,
signature and options. Primitive words, binary64-to-FP32 scale conversion,
local contraction routing and complete reduction trees are model data.
The numerical reference is the independent recurrence in KernelNative. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelCompiler
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelNativeCarrier
noncomputable section

inductive IntegerBinding where
  | runtime32 | runtime64 | specialized (value : Nat)
  deriving DecidableEq
def integerBinding : IntegerBinding → Nat → Prop
  | .runtime32,n => n<2^31
  | .runtime64,n => n<2^63
  | .specialized value,n => value=n
def integers (kind : Kind) (s : Shape) : List Nat :=
  match kind with
  | .forward => [s.T*s.K,s.T*s.V,s.B,s.H,s.T]
  | .backward => [s.T*s.K,s.T*s.V,1,s.B,s.H,s.T]
def pointerSlots (kind : Kind) (f : Flags) : List (Option RegionName) :=
  match kind with
  | .forward => [some "q",some "k",some "v",some "beta",some "o",
      if f.initial then some "h0" else none,if f.final then some "ht" else none]
  | .backward => [some "q",some "k",some "v",some "beta",
      if f.terminalGrad then some "dht" else none,if f.initialGrad then some "dh0" else none,
      some "do",some "dq",some "dk",some "dv",some "dbeta",if f.initial then some "h0" else none]
def scaleSlotIndex : Kind → Nat | .forward => 9 | .backward => 15
structure Signature where
  pointers : List (Option RegionName)
  pointerBits : Nat
  elementBits : Nat
  pointerDivisibility : List Nat
  integers : List IntegerBinding
  integerDivisibility : List Nat
  scaleFP32 : Bool
  scaleSlot : Nat
  K : Nat
  V : Nat
  BK : Nat
  BV : Nat
  initial : Bool
  final : Bool
  vectorBeta : Bool
  initialGrad : Bool
  terminalGrad : Bool
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
  conversion : String
structure Object where
  bytes : ByteArray
  sourceHash : String
  astHash : String
  entry : Kind
  signature : Signature
  options : Options
  primitives : PrimitiveNames
structure Layout where
  kind : Kind
  shape : Shape
  flags : Flags
  runtimeScale : Bool
  scaleArgument : Nat
  scaleMarker : ℝ
def layout (kind : Kind) (s : Shape) (f : Flags) : Layout :=
  ⟨kind,s,f,Bool.true,scaleSlotIndex kind,scaleSlot⟩
def template (recipe : Layout) : ComputeKernel := source recipe.kind recipe.shape recipe.flags recipe.scaleMarker
structure Decoder where
  layout : ByteArray → Signature → Options → List Nat → Layout
  routing : ByteArray → Signature → Options → PrimitiveNames → Routing
structure Platform where
  decoder : Decoder
  instructions : PrimitiveNames → Instructions
  scale : Object → UInt64 → Word
  defaultScale : Nat → UInt64
  defaults : Kind → Options
def astHash : Kind → String
  | .forward => "3a89e5082dc14071c9474788f11162f88be694739fa5993203a5043394bd6d31"
  | .backward => "a4b68eef374ce1cb8a089f48eb310a2806b5608db8475563c4988905993f32c3"
structure Binding (platform : Platform) (kind : Kind) (s : Shape) (f : Flags) where
  object : Object
  source : object.sourceHash=KernelProgram.sourceSHA256
  ast : object.astHash=astHash kind
  entry : object.entry=kind
  pointers : object.signature.pointers=pointerSlots kind f
  pointerBits : object.signature.pointerBits=64
  storage : object.signature.elementBits=32
  integers : List.Forall₂ integerBinding object.signature.integers (KernelCompiler.integers kind s)
  integerAttributes : List.Forall₂ (fun divisor value => 0<divisor ∧ value%divisor=0)
    object.signature.integerDivisibility (KernelCompiler.integers kind s)
  scale : object.signature.scaleFP32=Bool.true
  scalePosition : object.signature.scaleSlot=scaleSlotIndex kind
  dimensions : object.signature.K=s.K ∧ object.signature.V=s.V ∧ object.signature.BK=s.BK ∧ object.signature.BV=blockV kind s
  flags : object.signature.initial=f.initial ∧ object.signature.vectorBeta=f.vectorBeta ∧
    (kind=.forward → object.signature.final=f.final) ∧
    (kind=.backward → object.signature.initialGrad=f.initialGrad ∧ object.signature.terminalGrad=f.terminalGrad)
  options : object.options=platform.defaults kind
  backend : object.options.backend="cuda"
  version : object.options.compiler="3.7.1"
  warps : object.options.numWarps=(if kind=.forward then 1 else 2)
  stages : object.options.numStages=1
  ctas : object.options.numCTAs=1
  fusion : object.options.fusion=Bool.true
  decoded : platform.decoder.layout object.bytes object.signature object.options (KernelCompiler.integers kind s)=layout kind s f

def isa (platform : Platform) (binding : Binding platform kind s f) := platform.instructions binding.object.primitives
def routing (platform : Platform) (binding : Binding platform kind s f) :=
  platform.decoder.routing binding.object.bytes binding.object.signature binding.object.options binding.object.primitives
def algebra (platform : Platform) (binding : Binding platform kind s f) (host : UInt64) :=
  KernelNativeCarrier.algebra (isa platform binding) (routing platform binding) (platform.scale binding.object host)
def decodedSource (platform : Platform) (binding : Binding platform kind s f) :=
  template (platform.decoder.layout binding.object.bytes binding.object.signature binding.object.options (integers kind s))
theorem decoded_source (platform : Platform) (binding : Binding platform kind s f) :
    decodedSource platform binding=source kind s f scaleSlot := by
  unfold decodedSource
  rw [binding.decoded]
  rfl
theorem runtime_scale (platform : Platform) (binding : Binding platform kind s f) (host : UInt64) :
    ((algebra platform binding host).literal (some .fp32) .real scaleSlot).eval (isa platform binding) (routing platform binding)=
      platform.scale binding.object host := by
  simp [algebra,KernelNativeCarrier.algebra,Node.eval]

/-- An explicit constructor denotes a Python float argument. The absent
argument is evaluated by the platform's Python binary64 power operation. -/
inductive ScaleArgument where
  | default
  | explicitFloat (bits : UInt64)
def hostScale (platform : Platform) (s : Shape) : ScaleArgument → UInt64
  | .default => platform.defaultScale s.K
  | .explicitFloat bits => bits
def positiveFinite64 (bits : UInt64) : Prop := 0<bits.toNat ∧ bits.toNat<9218868437227405312
def finite32 (word : Word) : Prop := word.toNat/8388608%256≠255
def ScaleAdmitted (platform : Platform) (binding : Binding platform kind s f) (arg : ScaleArgument) : Prop :=
  positiveFinite64 (hostScale platform s arg) ∧ finite32 (platform.scale binding.object (hostScale platform s arg))
theorem positive_underflow_admitted (platform : Platform) (binding : Binding platform kind s f)
    (host : UInt64) (positive : positiveFinite64 host) (underflow : platform.scale binding.object host=0) :
    ScaleAdmitted platform binding (.explicitFloat host) := by
  refine ⟨positive,?_⟩
  simp [hostScale,underflow,finite32]

def SourceFacts (platform : Platform) (binding : Binding platform kind s f) (host : UInt64)
    (p : Program kind s) (u v : State Node) : Prop :=
  match kind with
  | .forward => KernelForward.Facts (algebra platform binding host) s f scaleSlot p.head p.value u v
  | .backward => KernelBackwardFacts.Facts (algebra platform binding host) s f scaleSlot p.head p.value u v
def Expected (platform : Platform) (binding : Binding platform kind s f) (host : UInt64)
    (p : Program kind s) (u : State Node) : RegionName → Nat → Word → Prop :=
  match kind with
  | .forward => KernelNative.ForwardExpected (isa platform binding) (routing platform binding)
      (platform.scale binding.object host) s f p.head p.value u
  | .backward => KernelNative.BackwardExpected (isa platform binding) (routing platform binding)
      (platform.scale binding.object host) s f p.head p.value u
structure Facts (platform : Platform) (binding : Binding platform kind s f) (host : UInt64)
    (p : Program kind s) (u v : State Node) : Prop where
  source : SourceFacts platform binding host p u v
  decodedSuccess : FP.Structural.exec (engine (algebra platform binding host))
    (decodedSource platform binding) u=some v
theorem complete (platform : Platform) (kind : Kind) (s : Shape) (f : Flags)
    (binding : Binding platform kind s f) (host : UInt64) (p : Program kind s) (u : State Node)
    (hv : u.pids 0=p.value) (hk : u.pids 1=0) (hh : u.pids 2=p.head) :
    ∃ v,Facts platform binding host p u v ∧
      ∀ r off word,Expected platform binding host p u r off word →
        observe (isa platform binding) (routing platform binding) (v.mem r off)=word := by
  cases kind with
  | forward =>
    obtain ⟨v,facts,values⟩ := KernelNative.forward_complete (isa platform binding) (routing platform binding)
      (platform.scale binding.object host) s f p.head p.value u hv hk hh
    refine ⟨v,⟨facts,?_⟩,values⟩
    rw [decoded_source]
    exact facts.success
  | backward =>
    obtain ⟨v,facts,values⟩ := KernelNative.backward_complete (isa platform binding) (routing platform binding)
      (platform.scale binding.object host) s f p.head p.value u hv hk hh
    refine ⟨v,⟨facts,?_⟩,values⟩
    rw [decoded_source]
    exact facts.success

#axiomsClean decoded_source
#axiomsClean runtime_scale
#axiomsClean positive_underflow_admitted
#axiomsClean complete
#specNonCircular Binding avoiding [FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelCompiler
