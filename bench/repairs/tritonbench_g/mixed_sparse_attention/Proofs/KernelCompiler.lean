import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelNativeReference
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.MixedSparseAttention.KernelCompiler
open VeriTile Triton FP.Structural Whole002 Source Whole Checked
noncomputable section
inductive IntegerABI where
  | i32 | i64 | constant (n : Nat)
def IntegerABI.binds : IntegerABI → Nat → Prop
  | .i32,n=>n<2^31 | .i64,n=>n<2^63 | .constant k,n=>k=n
structure IntegerArgument where
  kind : IntegerABI
  alignment : Nat
def integers (c : Config) : List Nat :=
  [c.sqz,c.sqh,c.sqm,c.sqk,c.skz,c.skh,c.skn,c.skk,c.svz,c.svh,c.svn,c.svk,
   c.soz,c.soh,c.som,c.sok,c.Z,c.H,c.T,c.NR,c.NS,c.NV]
structure Object where
  bytes : List (BitVec 8)
  sourceHash : String
  pointerFloat32 : Role → Bool
  pointerAlignment : Role → Nat
  integerArguments : List IntegerArgument
  constants : List Nat
  fp16Compute : Bool
  fp32Scale : Bool
  capability : Nat
  warps : Nat
  stages : Nat
/-- Explicit local lowering scheme: seeded m16n8k16 chains, sequential local
register reductions then XOR2/XOR1, and the source scalar/cast/FMA staging.
No final output, whole dot or whole-loop result occurs in this marker. -/
inductive LocalScheme where | sm80FourWarpFP16
structure Decoded where
  source : ComputeKernel
  scheme : LocalScheme
structure Platform where
  decode : Object → Option Decoded
  instructions : Object → Native.CompareISA (BitVec 32)
  bits : ∀ object word,(instructions object).bits word=word
def canonicalHash : String := "4b0bccf3382d289220accf0bc73fbfec200981ab136ad83b0bd0db16d545e553"
structure Binding (platform : Platform) (c : Config) (request : Request) where
  object : Object
  decoded : Decoded
  decoding : platform.decode object=some decoded
  quote : decoded.source=Source.kernel c
  scheme : decoded.scheme=.sm80FourWarpFP16
  hash : object.sourceHash=canonicalHash
  pointerTypes : ∀ r,object.pointerFloat32 r=r.floating
  pointers : ∀ r,0<object.pointerAlignment r ∧ (request.tensor r).pointer%object.pointerAlignment r=0
  integerArguments : List.Forall₂ (fun arg n=>arg.kind.binds n ∧ 0<arg.alignment ∧ n%arg.alignment=0)
    object.integerArguments (integers c)
  constants : object.constants=[c.BM,c.BN,c.BD]
  dtype : object.fp16Compute=Bool.true ∧ c.storage=.fp16
  scale : object.fp32Scale=Bool.true
  target : object.capability=80
  options : object.warps=4 ∧ object.stages=2
inductive ScaleArgument where
  | pythonFloat (binary64 : BitVec 64)
  | pythonInteger (value : Int)
structure ScaleConversion where
  convert : ScaleArgument → BitVec 32
def LaunchScale (conversion : ScaleConversion) (argument : ScaleArgument) (request : Request) : Prop :=
  conversion.convert argument=request.scaleWord

#specNonCircular Binding avoiding [FP.Structural.exec,Whole.runPrograms,Source.finalValues]
#specNonCircular LaunchScale avoiding [FP.Structural.exec,Whole.runPrograms,Source.finalValues]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.KernelCompiler
