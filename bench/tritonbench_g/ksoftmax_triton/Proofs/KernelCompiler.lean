import bench.tritonbench_g.ksoftmax_triton.Proofs.Graph
import bench.tritonbench_g.ksoftmax_triton.Proofs.Pointers

/-! Input-independent compiler/object/ABI metadata for the selected source.
The decoder exposes only instruction choices, reduction structure, types,
addresses and launch constants. It cannot return an output or an execution.
-/
namespace VeriTile.Bench.Repairs.Ksoftmax.KernelCompiler
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

def sourceSHA256 : String := "03d1517e3ac3ef0fdab20673a410d248d11461c1ddbc12df9e262119a98bd1e4"

def leaves {ι : Type} : Graph.Plan ι → List ι
  | .lane i => [i]
  | .constant _ => []
  | .add a b | .max a b | .fmaLeft a b | .fmaRight a b => leaves a++leaves b

def sumTree {ι : Type} : Graph.Plan ι → Prop
  | .lane _ => True
  | .constant word => word=Graph.zero
  | .add a b | .fmaLeft a b | .fmaRight a b => sumTree a ∧ sumTree b
  | .max _ _ => False

def maxTree {ι : Type} : Graph.Plan ι → Prop
  | .lane _ => True
  | .constant word => word=Graph.negInf
  | .max a b => maxTree a ∧ maxTree b
  | _ => False

/-- Every actual row-reduction lane occurs exactly once. Ordered binary/FMA
trees retain reduction order; no arbitrary constants or duplicate/omitted lanes
can be used as a supposed selected reduction. -/
structure ValidReductions (profile : Graph.Profile) (D : Nat) : Prop where
  sum : sumTree (profile.plans.sum (shape := [D]) ⟨0,by simp⟩ Bool.false PUnit.unit)
  maximum : maxTree (profile.plans.maximum (shape := [D]) ⟨0,by simp⟩ Bool.false PUnit.unit)
  sumLanes : (leaves (profile.plans.sum (shape := [D]) ⟨0,by simp⟩ Bool.false PUnit.unit)).Perm
    ((List.finRange D).map fun i => (i,PUnit.unit))
  maxLanes : (leaves (profile.plans.maximum (shape := [D]) ⟨0,by simp⟩ Bool.false PUnit.unit)).Perm
    ((List.finRange D).map fun i => (i,PUnit.unit))

structure Signature where
  pointers : List (BitVec 64)
  pointeeWidths : List Nat
  integerWidths : List Nat
  integerArguments : List (BitVec 64)
  depth : Nat
  operation : Operation
  fp16 : Bool

structure Options where
  numWarps : Nat
  enableFPFusion : Bool
  deriving DecidableEq

structure Primitives where
  add : String
  sub : String
  mul : String
  fma : String
  division : String
  maximum : String
  exp2 : String
  logarithm : String

structure Object where
  bytes : ByteArray
  sourceHash : String
  signature : Signature
  options : Options
  primitives : Primitives

/-- Source address recipes name the actual ABI pointer/scalar slots. -/
inductive AddressRecipe where
  | matrix (pointerSlot batchStrideSlot rowStrideSlot : Nat)
  | mask (pointerSlot rowProgramAxis strideSlot : Nat)
  deriving DecidableEq

def addresses : Operation → List AddressRecipe
  | .forward _ _ mode => [.matrix 1 2 3]++
      (if mode.present then [.mask 2 (if mode=.qk then 1 else 0) 4] else [])++[.matrix 0 0 1]
  | .backward _ _ => [.matrix 1 2 3,.matrix 2 4 5,.matrix 0 0 1]

structure Lowering where
  profile : Graph.Profile
  operation : Operation
  integerWidth : Nat
  fp32Loads : Bool
  fp32Arithmetic : Bool
  fp32Stores : Bool
  addresses : List AddressRecipe
  finalCausalGradientZero : Bool

structure Platform where
  decode : ByteArray → Signature → Options → Primitives → Option Lowering
  instructions : Primitives → Graph.Operations

def integerArguments (op : Operation) (Q K : Nat) : List Nat :=
  match op with
  | .forward _ _ mode => [Q*K,K,Q*K,K,if mode.present then K else 0,K]
  | .backward _ _ => [Q*K,K,Q*K,K,Q*K,K,K]

structure Selected (platform : Platform) (op : Operation) (Q K D : Nat)
    (dst left right : TensorMetadata) where
  object : Object
  lowering : Lowering
  source : object.sourceHash=sourceSHA256
  decoded : platform.decode object.bytes object.signature object.options object.primitives=some lowering
  operation : lowering.operation=op
  scalarABI : object.signature.integerArguments=(integerArguments op Q K).map (BitVec.ofNat 64)
  scalarWidths : object.signature.integerWidths=List.replicate (integerArguments op Q K).length 64
  pointers : object.signature.pointers=[dst.dataPtr,left.dataPtr,right.dataPtr]
  pointeeWidths : object.signature.pointeeWidths=[32,32,32]
  branch : object.signature.operation=op
  fp32 : object.signature.fp16=Bool.false
  depth : object.signature.depth=D
  depthPowerOfTwo : ∃ exponent : Nat,D=2^exponent
  profile : Graph.Binding lowering.profile K D
  reductions : ValidReductions lowering.profile D
  options : object.options={numWarps := lowering.profile.numWarps,enableFPFusion := Bool.true}
  precision : lowering.integerWidth=64 ∧ lowering.fp32Loads=Bool.true ∧
    lowering.fp32Arithmetic=Bool.true ∧ lowering.fp32Stores=Bool.true
  addresses : lowering.addresses=KernelCompiler.addresses op
  causalRepair : lowering.finalCausalGradientZero=Bool.true

abbrev Selected.ops (code : Selected platform op Q K D dst left right) :=
  platform.instructions code.object.primitives

abbrev Selected.model (code : Selected platform op Q K D dst left right) (rest : Algebra Graph.Expr) :=
  Graph.algebra rest code.lowering.profile

theorem abi_scalar_values (x : TensorMetadata) (rank : x.shape.length=3)
    (extent : ExtentContract x) (live : 0<x.numel) (op : Operation) :
    ∀ v ∈ integerArguments op (rows x) (columns x),
      VeriTile.Bench.Repairs.VarLenCopy.signed64 (v : Int)=(v : Int) := by
  obtain ⟨hk,hq⟩ := Pointers.strides x rank extent live
  have product := hq
  simp only [Nat.cast_mul] at product
  have zero : VeriTile.Bench.Repairs.VarLenCopy.signed64 (0 : Int)=0 := by decide
  cases op with
  | forward log causal mode =>
    cases mode <;> simp [integerArguments,Forward.MaskMode.present,hk,product,zero]
  | backward log causal => simp [integerArguments,hk,product]

#specNonCircular ValidReductions avoiding [forward,backward,runRows,Graph.eval]
#specNonCircular Selected avoiding [forward,backward,runRows,Graph.eval]
#axiomsClean abi_scalar_values
end
end VeriTile.Bench.Repairs.Ksoftmax.KernelCompiler
