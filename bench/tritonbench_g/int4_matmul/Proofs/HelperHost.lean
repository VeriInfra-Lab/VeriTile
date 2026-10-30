import bench.tritonbench_g.int4_matmul.Proofs.Host
import bench.tritonbench_g.int4_matmul.Proofs.HelperNative
namespace VeriTile.Bench.Repairs.Int4Matmul.HelperHost
open Host Helpers Physical HelperBytes
/-- Fresh returned allocations are ordinary allocator contracts. -/
structure QuantAllocation (s : Shape) (device input : Nat) where
  weights : Tensor
  scales : Tensor
  zeros : Tensor
  weightLayout : weights.TransposeContiguous .int32 (s.k/8) s.n device
  scaleLayout : scales.TransposeContiguous .fp32 (s.k/s.group) s.n device
  zeroLayout : zeros.TransposeContiguous .int32 (s.k/s.group) (s.n/8) device
  independent : [input,weights.allocation,scales.allocation,zeros.allocation].Nodup

structure UnpackAllocation (s : Shape) (device : Nat) (callerAllocations : List Nat) where
  tensor : Tensor
  layout : tensor.rows=s.k ∧ tensor.columns=s.n ∧ tensor.rowStride=1 ∧ tensor.columnStride=s.k ∧
    tensor.dtype=.fp16 ∧ tensor.device=device ∧ tensor.offset=0 ∧ tensor.lazyNeg=Bool.false ∧ tensor.lazyConj=Bool.false ∧ tensor.requiresGrad=Bool.false
  fresh : tensor.allocation∉callerAllocations

/-- The finite and positive-scale guards have explicit binary32 meanings. -/
def finite32 (w : Word) : Prop := ((w >>> 23) &&& (255:Word)).toNat≠255
def positive32 (w : Word) : Prop := w.toNat<2^31 ∧ w.toNat≠0 ∧ finite32 w
structure QuantDomain (s : Shape) (p : HelperNative.Instructions) (order : ReductionOrder s.group)
    (initial : Bytes) (weight : Tensor) (tpRank : Nat) : Prop where
  shape : s.Valid
  layout : weight.Canonical .fp32 s.k s.n weight.device
  rank : tpRank=weight.device
  indices : s.k*s.n<2^31
  finite : ∀ k n,k<s.k → n<s.n → finite32 ((load initial).values .weight (k*s.n+n))
  positiveScale : ∀ g n,g<s.k/s.group → n<s.n →
    positive32 (scale s (HelperNative.arithmetic p) order (load initial) g n)

/-- The actual unpack wrapper's ordinary packed input admission. -/
structure PackedDomain (s : Shape) (weight scales zeros : Tensor) : Prop where
  shape : s.Valid
  weightLayout : weight.PackedLayout .int32 (s.k/8) s.n weight.device
  scale : scales.PackedLayout .fp32 (s.k/s.group) s.n weight.device
  zero : zeros.PackedLayout .int32 (s.k/s.group) (s.n/8) weight.device
  independent : [weight.allocation,scales.allocation,zeros.allocation].Nodup
  indices : s.k*s.n<2^31

structure QuantReturn where
  weights : Tensor
  scales : Tensor
  zeros : Tensor
  group : Nat
  data : QuantizedBytes
  callerDevice : Nat

def quantCall (s : Shape) (p : HelperNative.Instructions) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) (device input caller : Nat)
    (allocation : QuantAllocation s device input) : QuantReturn :=
  ⟨allocation.weights,allocation.scales,allocation.zeros,s.group,
    quantizeBytes s (HelperNative.arithmetic p) order initial scratch,caller⟩


/-- The returned descriptors address exactly the serialized row-major payload,
including each retained singleton stride. -/
structure QuantAddresses (s : Shape) (result : QuantReturn) : Prop where
  weights : ∀ p n,p<s.k/8 → n<s.n → result.weights.elementOffset p n=p*s.n+n
  scales : ∀ g n,g<s.k/s.group → n<s.n → result.scales.elementOffset g n=g*s.n+n
  zeros : ∀ g p,g<s.k/s.group → p<s.n/8 → result.zeros.elementOffset g p=g*(s.n/8)+p

theorem quant_addresses (s : Shape) (p : HelperNative.Instructions) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) (device input caller : Nat)
    (a : QuantAllocation s device input) :
    QuantAddresses s (quantCall s p order initial scratch device input caller a) := by
  refine ⟨?_,?_,?_⟩
  · intro r c hr hc
    exact a.weights.packed_active_offset _ _ _ _ _ _ (Or.inr a.weightLayout) hr hc
  · intro r c hr hc
    exact a.scales.packed_active_offset _ _ _ _ _ _ (Or.inr a.scaleLayout) hr hc
  · intro r c hr hc
    exact a.zeros.packed_active_offset _ _ _ _ _ _ (Or.inr a.zeroLayout) hr hc

/-- The returned triple satisfies the successor's packed-layout admission. -/
theorem quant_packed_layout (s : Shape) (device input : Nat) (a : QuantAllocation s device input) :
    a.weights.PackedLayout .int32 (s.k/8) s.n device ∧
    a.scales.PackedLayout .fp32 (s.k/s.group) s.n device ∧
    a.zeros.PackedLayout .int32 (s.k/s.group) (s.n/8) device :=
  ⟨Or.inr a.weightLayout,Or.inr a.scaleLayout,Or.inr a.zeroLayout⟩

#axiomsClean quant_addresses
#axiomsClean quant_packed_layout

structure UnpackReturn where
  tensor : Tensor
  data : Unpacked
  callerDevice : Nat

def unpackCall (s : Shape) (p : HelperNative.Instructions) (initial : Bytes)
    (scratch : Nat → Byte) (device caller : Nat) (callerAllocations : List Nat) (allocation : UnpackAllocation s device callerAllocations) : UnpackReturn :=
  ⟨allocation.tensor,unpack s (HelperNative.halfArithmetic p) initial scratch,caller⟩

theorem quant_return (s : Shape) (p : HelperNative.Instructions) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) (device input caller : Nat)
    (a : QuantAllocation s device input) :
    (quantCall s p order initial scratch device input caller a).weights=a.weights ∧
    (quantCall s p order initial scratch device input caller a).scales=a.scales ∧
    (quantCall s p order initial scratch device input caller a).zeros=a.zeros ∧
    (quantCall s p order initial scratch device input caller a).group=s.group ∧
    (quantCall s p order initial scratch device input caller a).callerDevice=caller := ⟨rfl,rfl,rfl,rfl,rfl⟩

theorem unpack_return (s : Shape) (p : HelperNative.Instructions) (initial : Bytes)
    (scratch : Nat → Byte) (device caller : Nat) (callerAllocations : List Nat) (a : UnpackAllocation s device callerAllocations) :
    (unpackCall s p initial scratch device caller callerAllocations a).tensor=a.tensor ∧
    (unpackCall s p initial scratch device caller callerAllocations a).callerDevice=caller := ⟨rfl,rfl⟩
#axiomsClean quant_return
#axiomsClean unpack_return
end VeriTile.Bench.Repairs.Int4Matmul.HelperHost
