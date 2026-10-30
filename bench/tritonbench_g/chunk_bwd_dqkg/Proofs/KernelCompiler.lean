import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelNativeReference
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelTensor

/-! Local compiled-object interpretation, not an output-correctness premise.
The decoder consumes only object bytes/signature/options. Its result contains
source syntax and value-erased instruction routing. The SM90 K8/scalar ports
and numeric-to-FP32 conversion are explicitly the primitive boundary. -/
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelCompiler
open VeriTile.Triton Whole KernelNativeCarrier KernelContract
noncomputable section
inductive IntegerBinding where
  | i32 | i64 | constant (n : Nat)
  deriving DecidableEq
def IntegerBinding.binds : IntegerBinding → Nat → Prop
  | .i32,n=>n<2^31 | .i64,n=>n<2^63 | .constant k,n=>k=n
structure ScalarBinding where
  kind : IntegerBinding
  alignment : Nat
def integerValues (c : Config) : List Nat :=
  [c.T*c.K,c.K,c.T*c.V,c.V,c.NT*c.K*c.V,c.V]
def pointerOrder : List Role := [.q,.k,.v,.h,.g,.dout,.dh,.dq,.dk,.dg]
def constexprValues (c : Config) : List Nat := [c.T,c.K,c.V,64,c.BK,c.BV,c.NT]
structure Object where
  bytes : List (BitVec 8)
  sourceHash : String
  pointerAlignment : Role → Nat
  integers : List ScalarBinding
  constants : List Nat
  scaleFP32 : Bool
  warps : Nat
  stages : Nat
  capability : Nat
structure Decoded where
  source : ComputeKernel
  routing : Routing
structure Platform where
  decode : Object → Option Decoded
  instructions : Object → Instructions
inductive ScaleArgument where
  | pythonFloat (binary64 : BitVec 64)
  | pythonInteger (value : Int)
structure ScaleConversion where
  convert : ScaleArgument → BitVec 32
def sourceHash : String := "7c2c6aef92283c9c3f9f4980291ab2367eaf6096436b5847c683a560335a1ac6"
structure Binding (platform : Platform) (b : KernelTensor.Buffers c) where
  object : Object
  decoded : Decoded
  decoding : platform.decode object=some decoded
  quote : decoded.source=Whole.kernel (nativeConfig c)
  hash : object.sourceHash=sourceHash
  integerABI : List.Forall₂ (fun arg n=>arg.kind.binds n ∧ 0<arg.alignment ∧ n%arg.alignment=0)
    object.integers (integerValues c)
  pointerABI : ∀ r,0<object.pointerAlignment r ∧
    (b.tensor (region r)).metadata.dataPtr.toNat%object.pointerAlignment r=0
  constexprs : object.constants=constexprValues c
  scaleType : object.scaleFP32=Bool.true
  warps : object.warps=4 ∨ object.warps=8
  stages : object.stages=3
  target : object.capability=90
structure LaunchScale (conversion : ScaleConversion) (argument : ScaleArgument) (c : Config) where
  value : Scale
  word : value.word.bits=conversion.convert argument
  real : value.real=c.scale

theorem source_decoded (platform : Platform) (b : KernelTensor.Buffers c) (binding : Binding platform b) :
    binding.decoded.source=Whole.kernel (nativeConfig c) := binding.quote

#axiomsClean source_decoded
#specNonCircular Binding avoiding [Typed.runGrid,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelCompiler
