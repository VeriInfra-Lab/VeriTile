import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.Strides
import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.Pipeline
import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.Native
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Host
set_option maxHeartbeats 4000000
noncomputable section
inductive DType where
  | fp16 | fp32 | int32
  deriving DecidableEq, Repr
def DType.width : DType → Nat
  | .fp16 => 2
  | .fp32 | .int32 => 4
structure Tensor where
  objectId : Nat
  base : Nat
  storageBytes : Nat
  rows : Nat
  columns : Nat
  strides : Strides.Layout
  dtype : DType
  device : Nat
  offset : Nat
  lazyNeg : Bool
  lazyConj : Bool
  requiresGrad : Bool
  deriving DecidableEq, Repr

/-- Exact ordinary Tensor/type/rank/CUDA admission is the host ABI boundary.
Storage capacity and disjoint byte intervals have no numerical payload premises. -/
def Tensor.Ordinary (x : Tensor) (dtype : DType) (rows columns device : Nat) :=
  x.rows=rows ∧ x.columns=columns ∧ 0<rows ∧ 0<columns ∧
  x.strides.Admitted rows columns ∧ x.dtype=dtype ∧ x.device=device ∧
  x.offset=0 ∧ x.lazyNeg=false ∧ x.lazyConj=false ∧ x.requiresGrad=false ∧
  dtype.width*rows*columns≤x.storageBytes ∧ x.base+x.storageBytes<2^64
def Tensor.FreshLayout (x : Tensor) (dtype : DType) (rows columns device : Nat) :=
  x.Ordinary dtype rows columns device ∧ x.strides=Strides.canonical columns
def Tensor.QuantLayout (x : Tensor) (dtype : DType) (rows columns device : Nat) :=
  x.Ordinary dtype rows columns device ∧ x.strides=Strides.quantReturned rows columns
def Disjoint (x y : Tensor) := x.base+x.storageBytes≤y.base ∨ y.base+y.storageBytes≤x.base
def Independent (xs : List Tensor) := xs.Pairwise Disjoint
theorem disjoint_byte_addresses (x y : Tensor) (h : Disjoint x y) (a b : Nat)
    (ha : a<x.storageBytes) (hb : b<y.storageBytes) : x.base+a≠y.base+b := by
  rcases h with h|h <;> omega
theorem ordinary_active (x : Tensor) (dtype : DType) (rows columns device r c : Nat)
    (h : x.Ordinary dtype rows columns device) (hr : r<rows) (hc : c<columns) :
    x.strides.address r c=r*columns+c := Strides.active_address _ _ _ _ _ h.2.2.2.2.1 hr hc

structure PackedArguments where
  shape : Shape
  weight : Tensor
  scale : Tensor
  zero : Tensor
def PackedArguments.strides (x : PackedArguments) : Strides.Packed :=
  ⟨x.weight.strides,x.scale.strides,x.zero.strides⟩
structure PackedContract (x : PackedArguments) (dtype : DType) : Prop where
  selected : dtype=.fp16 ∨ dtype=.fp32
  shape : x.shape.Valid
  weight : x.weight.Ordinary .int32 (x.shape.k/8) x.shape.n x.weight.device
  scale : x.scale.Ordinary dtype (x.shape.k/x.shape.group) x.shape.n x.weight.device
  zero : x.zero.Ordinary .int32 (x.shape.k/x.shape.group) (x.shape.n/8) x.weight.device
  independent : Independent [x.weight,x.scale,x.zero]
  indices : Strides.Guard x.strides x.shape
theorem packed_admitted (x : PackedArguments) (dtype : DType) (h : PackedContract x dtype) :
    x.strides.Admitted x.shape :=
  ⟨h.weight.2.2.2.2.1,h.scale.2.2.2.2.1,h.zero.2.2.2.2.1⟩

structure Arguments where
  packed : PackedArguments
  input : Tensor
  supplied : Option Tensor
def Arguments.output (x : Arguments) (fresh : Tensor) := x.supplied.getD fresh
structure Contract (x : Arguments) : Prop where
  packed : PackedContract x.packed .fp16
  input : x.input.Ordinary .fp16 x.packed.shape.m x.packed.shape.k x.packed.weight.device
  independent : Independent [x.input,x.packed.weight,x.packed.scale,x.packed.zero]
  supplied : ∀ out,x.supplied=some out →
    out.Ordinary .fp16 x.packed.shape.m x.packed.shape.n x.packed.weight.device ∧
    (∀ t∈[x.input,x.packed.weight,x.packed.scale,x.packed.zero],Disjoint out t)
  inputIndices : x.packed.shape.m*x.packed.shape.k<2^31
  outputIndices : x.packed.shape.m*x.packed.shape.n<2^31

/-- Successful allocator calls supply fresh storage, with actual source layouts.
Allocator/OOM failure and asynchronous device faults are outside successful calls. -/
structure Allocations (x : Arguments) where
  freshOutput : Tensor
  scratch : Tensor
  freshLayout : freshOutput.FreshLayout .fp16 x.packed.shape.m x.packed.shape.n x.packed.weight.device
  scratchLayout : scratch.FreshLayout .fp16 x.packed.shape.k x.packed.shape.n x.packed.weight.device
  freshDisjoint : ∀ t∈[x.input,x.packed.weight,x.packed.scale,x.packed.zero],Disjoint freshOutput t
  scratchDisjoint : ∀ t∈[x.input,x.packed.weight,x.packed.scale,x.packed.zero,x.output freshOutput],Disjoint scratch t

structure Return where
  tensor : Tensor
  data : Physical.Bytes
  launchDevice : Nat
  callerDevice : Nat
def call (x : Arguments) (allocation : Allocations x) (p : Native.Instructions)
    (plan : Library.Selected x.packed.shape) (trials : List (List Work))
    (ds os : List Work) (initial : Physical.Bytes) (caller : Nat) : Return :=
  ⟨x.output allocation.freshOutput,
   Pipeline.s1Bytes Physical.halfCodec x.packed.shape 0x3c00 (Native.dequant16 p) (Native.library p)
     plan trials ds os initial,x.packed.weight.device,caller⟩

theorem supplied_identity (x : Arguments) (out fresh : Tensor) (h : x.supplied=some out) :
    x.output fresh=out := by simp [Arguments.output,h]
theorem fresh_identity (x : Arguments) (fresh : Tensor) (h : x.supplied=none) :
    x.output fresh=fresh := by simp [Arguments.output,h]
theorem return_geometry (x : Arguments) (h : Contract x) (allocation : Allocations x) :
    (x.output allocation.freshOutput).Ordinary .fp16 x.packed.shape.m x.packed.shape.n x.packed.weight.device := by
  cases ho : x.supplied with
  | none => simpa [Arguments.output,ho] using allocation.freshLayout.1
  | some out => simpa [Arguments.output,ho] using (h.supplied out ho).1
theorem returned_bytes (x : Arguments) (allocation : Allocations x) (p : Native.Instructions)
    (plan : Library.Selected x.packed.shape) (trials : List (List Work))
    (dg : Grid x.packed.shape.k x.packed.shape.n) (og : Grid x.packed.shape.m x.packed.shape.n)
    (ds os : List Work) (hd : dg.Complete ds) (ho : og.Complete os) (initial : Physical.Bytes)
    (caller r n : Nat) (hr : r<x.packed.shape.m) (hn : n<x.packed.shape.n) (i : Fin 2) :
    (call x allocation p plan trials ds os initial caller).data (.value .output) (2*(r*x.packed.shape.n+n)+i.val)=
      Physical.encode16 (S1.reference x.packed.shape (Native.dequant16 p) (Native.library p) plan
        (Physical.load Physical.halfCodec initial) r n) i :=
  Pipeline.s1_result Physical.halfCodec x.packed.shape 0x3c00 (Native.dequant16 p) (Native.library p)
    plan trials dg og ds os hd ho initial r n hr hn i
theorem device_restored (x : Arguments) (allocation : Allocations x) (p : Native.Instructions)
    (plan : Library.Selected x.packed.shape) (trials : List (List Work))
    (ds os : List Work) (initial : Physical.Bytes) (caller : Nat) :
    (call x allocation p plan trials ds os initial caller).callerDevice=caller ∧
    (call x allocation p plan trials ds os initial caller).launchDevice=x.packed.weight.device := ⟨rfl,rfl⟩

/-- Ordinary Python with-context exit semantics, including a raised body error.
The library/runtime supplies that context-manager behavior; no GPU result is assumed. -/
def deviceScope (caller target : Nat) (body : Nat → Except ε α) : Except ε α × Nat := (body target,caller)
theorem context_exit_restores (caller target : Nat) (body : Nat → Except ε α) :
    (deviceScope caller target body).2=caller := rfl

#axiomsClean disjoint_byte_addresses
#axiomsClean ordinary_active
#axiomsClean packed_admitted
#axiomsClean supplied_identity
#axiomsClean fresh_identity
#axiomsClean return_geometry
#axiomsClean returned_bytes
#axiomsClean device_restored
#axiomsClean context_exit_restores
end
end VeriTile.Bench.Repairs.MatmulDequantInt4.Host
