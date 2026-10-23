import bench.repairs.tritonbench_g.int4_matmul.Proofs.DirectMatmulNative
import bench.repairs.tritonbench_g.int4_matmul.Proofs.DirectQuantNative
import bench.repairs.tritonbench_g.int4_matmul.Proofs.DirectUnpack
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Real
import bench.repairs.tritonbench_g.int4_matmul.Proofs.HelperReal
import bench.repairs.tritonbench_g.int4_matmul.Proofs.HelperBounds
import bench.repairs.tritonbench_g.int4_matmul.Proofs.HelperHost
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Source
namespace VeriTile.Bench.Repairs.Int4Matmul
open Host Helpers Physical HelperBytes HelperHost

structure NativeReturn where
  tensor : Tensor
  payload : Bytes
  callerDevice : Nat

def selected (x : Arguments) (t : Tile) (p : Native.Instructions) (isa : Native.ISA)
    (ws : List Work) (initial : Bytes) (fresh : Tensor) (caller : Nat) : NativeReturn :=
  ⟨outputTensor x fresh,byteCall x.shape t p isa ws initial,caller⟩

structure MatmulNativeClaim (x : Arguments) (t : Tile) (actual reference : Native.Instructions)
    (isa : Native.ISA) (ws : List Work) (initial : Bytes) (fresh : Tensor) (caller : Nat)
    (result : NativeReturn) : Prop where
  identity : result.tensor=outputTensor x fresh
  layout : result.tensor.Canonical .fp32 x.shape.m x.shape.n x.input.device
  device : result.callerDevice=caller
  source : load result.payload=Source.prepared x.shape t (Native.ops actual isa) (load initial) ws
    (zeroOutput x.shape (Native.ops actual isa) (load initial))
  values : ∀ r c,r<x.shape.m → c<x.shape.n →
    (load result.payload).values .output (cAddress x.shape r c)=
      referenceOutput x.shape t (Native.ops reference isa) (load initial) r c ws 0
  bytes : ∀ r c,r<x.shape.m → c<x.shape.n → ∀ i : Fin 4,
    result.payload (.value .output) (4*cAddress x.shape r c+i.val)=
      encode (referenceOutput x.shape t (Native.ops reference isa) (load initial) r c ws 0) i
  frame : ∀ q b,q≠.value .output ∨ x.shape.m*x.shape.n≤b/4 → result.payload q b=initial q b
  indices : IndexDomain x.shape t
  actualStrides : StrideBinding x t
  grid : blocksM x.shape t*blocksN x.shape t<2^31 ∧ t.gm*blocksN x.shape t<2^31

theorem matmul_native (x : Arguments) (valid : Contract x) (t : Tile) (member : t∈sourceTiles)
    (p : Native.Instructions) (isa : Native.ISA) (ws : List Work) (complete : Complete x.shape t ws)
    (initial : Bytes) (fresh : Tensor) (layout : FreshOutput x fresh)
    (caller : Nat) : MatmulNativeClaim x t p p isa ws initial fresh caller (selected x t p isa ws initial fresh caller) := by
  have ht := all_tiles_valid t member
  refine ⟨rfl,return_geometry x valid fresh layout.layout,rfl,?_,?_,?_,?_,
    guarded_indices x.shape valid.indices t member,contract_strides x valid t member,grid_indices x.shape valid.indices t member⟩
  · change load (byteCall x.shape t p isa ws initial)=_
    rw [call_load]
    exact (Source.prepared_execute x.shape t (Native.ops p isa) (load initial)
      (zeroOutput x.shape (Native.ops p isa) (load initial)) ws
      (fun w hw => (complete_active x.shape t valid.shape ht ws complete w).mp hw)
      (zero_readonly x.shape (Native.ops p isa) (load initial))).symm
  · intro r c hr hc
    change (load (byteCall x.shape t p isa ws initial)).values .output (cAddress x.shape r c)=_
    rw [call_load]
    exact call_result x.shape t valid.shape ht (Native.ops p isa) ws complete (load initial) r c hr hc
  · exact call_bytes x.shape t valid.shape ht p isa ws complete initial
  · exact call_frame_bytes x.shape t valid.shape ht p isa ws complete initial

structure QuantNativeClaim (s : Shape) (reference : HelperNative.Instructions)
    (order : ReductionOrder s.group) (initial : Bytes) (device input caller : Nat)
    (allocation : QuantAllocation s device input) (result : QuantReturn) : Prop where
  identity : result.weights=allocation.weights ∧ result.scales=allocation.scales ∧ result.zeros=allocation.zeros
  layout : result.weights.TransposeContiguous .int32 (s.k/8) s.n device ∧
    result.scales.TransposeContiguous .fp32 (s.k/s.group) s.n device ∧ result.zeros.TransposeContiguous .int32 (s.k/s.group) (s.n/8) device
  addresses : QuantAddresses s result
  group : result.group=s.group
  device : result.callerDevice=caller
  callerFrame : result.data.caller=initial
  weights : ∀ p n,p<s.k/8 → n<s.n → ∀ i : Fin 4,
    result.data.weights (4*(p*s.n+n)+i.val)=encode (weightSpec s (HelperNative.arithmetic reference) order (load initial) p n) i
  scales : ∀ g n,g<s.k/s.group → n<s.n → ∀ i : Fin 4,
    result.data.scales (4*(g*s.n+n)+i.val)=encode (scale s (HelperNative.arithmetic reference) order (load initial) g n) i
  zeros : ∀ g p,g<s.k/s.group → p<s.n/8 → ∀ i : Fin 4,
    result.data.zeros (4*(g*(s.n/8)+p)+i.val)=encode (zeroSpec s (HelperNative.arithmetic reference) order (load initial) g p) i
  fresh : [input,result.weights.allocation,result.scales.allocation,result.zeros.allocation].Nodup

theorem quantize_native (s : Shape) (hs : s.Valid) (p : HelperNative.Instructions)
    (order : ReductionOrder s.group) (initial : Bytes) (scratch : Fin 3 → Nat → Byte)
    (device input caller : Nat) (a : QuantAllocation s device input) :
    QuantNativeClaim s p order initial device input caller a (quantCall s p order initial scratch device input caller a) := by
  exact ⟨⟨rfl,rfl,rfl⟩,⟨a.weightLayout,a.scaleLayout,a.zeroLayout⟩,quant_addresses s p order initial scratch device input caller a,rfl,rfl,rfl,
    quantize_weight_bytes s hs _ order initial scratch,
    quantize_scale_bytes s hs _ order initial scratch,
    quantize_zero_bytes s hs _ order initial scratch,a.independent⟩

structure UnpackNativeClaim (s : Shape) (reference : HelperNative.Instructions) (initial : Bytes)
    (device caller : Nat) (inputs : List Nat) (allocation : UnpackAllocation s device inputs)
    (result : UnpackReturn) : Prop where
  identity : result.tensor=allocation.tensor
  layout : result.tensor.rows=s.k ∧ result.tensor.columns=s.n ∧ result.tensor.rowStride=1 ∧ result.tensor.columnStride=s.k ∧
    result.tensor.dtype=.fp16 ∧ result.tensor.device=device ∧ result.tensor.offset=0 ∧ result.tensor.lazyNeg=Bool.false ∧
    result.tensor.lazyConj=Bool.false ∧ result.tensor.requiresGrad=Bool.false
  device : result.callerDevice=caller
  callerFrame : result.data.caller=initial
  bytes : ∀ k n,k<s.k → n<s.n → ∀ i : Fin 2,
    result.data.payload (2*(n*s.k+k)+i.val)=encode16 (unpackSpec s (HelperNative.halfArithmetic reference) (load initial) k n) i
  fresh : result.tensor.allocation∉inputs

theorem unpack_native (s : Shape) (hs : s.Valid) (p : HelperNative.Instructions) (initial : Bytes)
    (scratch : Nat → Byte) (device caller : Nat) (inputs : List Nat) (a : UnpackAllocation s device inputs) :
    UnpackNativeClaim s p initial device caller inputs a (unpackCall s p initial scratch device caller inputs a) :=
  ⟨rfl,a.layout,rfl,rfl,unpack_bytes s hs _ initial scratch,a.fresh⟩

/-- Whole task native contract, with separate complete statements for GEMM,
quantizer and explicit-half unpack helper. No API result is a numerical port. -/
structure WholeNative (actual reference : Native.Instructions) (helper helperReference : HelperNative.Instructions) : Prop where
  matmul : ∀ x,Contract x → ∀ t,t∈sourceTiles → ∀ isa ws,Complete x.shape t ws → ∀ initial fresh,
    FreshOutput x fresh → ∀ caller,
    MatmulNativeClaim x t actual reference isa ws initial fresh caller (selected x t actual isa ws initial fresh caller)
  quantizer : ∀ s,∀ order initial weight tpRank,QuantDomain s helper order initial weight tpRank →
    ∀ scratch caller (a : QuantAllocation s weight.device weight.allocation),
    QuantNativeClaim s helperReference order initial weight.device weight.allocation caller a
      (quantCall s helper order initial scratch weight.device weight.allocation caller a)
  unpacker : ∀ s weight scales zeros,PackedDomain s weight scales zeros →
    ∀ initial scratch caller (a : UnpackAllocation s weight.device [weight.allocation,scales.allocation,zeros.allocation]),
    UnpackNativeClaim s helperReference initial weight.device caller [weight.allocation,scales.allocation,zeros.allocation] a
      (unpackCall s helper initial scratch weight.device caller [weight.allocation,scales.allocation,zeros.allocation] a)

theorem whole_native (matrix : Native.Instructions) (helper : HelperNative.Instructions) :
    WholeNative matrix matrix helper helper := by
  refine ⟨?_,?_,?_⟩
  · intro x hx t ht isa ws hw initial fresh hf caller
    exact matmul_native x hx t ht matrix isa ws hw initial fresh hf caller
  · intro s order initial weight tpRank domain scratch caller a
    exact quantize_native s domain.shape helper order initial scratch weight.device weight.allocation caller a
  · intro s weight scales zeros domain initial scratch caller a
    exact unpack_native s domain.shape helper initial scratch weight.device caller _ a

/-- Four matrix fields (including fixed K=8 ISA), and thirteen scalar helper
fields. There is no whole-dot, whole-output, frame or source-equivalence premise. -/
theorem whole_fp (actual reference : Native.Instructions) (helper helperReference : HelperNative.Instructions)
    (matrixNumeric : Native.Correspondence actual reference)
    (helperNumeric : HelperNative.Correspondence helper helperReference) :
    WholeNative actual reference helper helperReference := by
  rw [←Native.correspondence_eq actual reference matrixNumeric,
    ←HelperNative.correspondence_eq helper helperReference helperNumeric]
  exact whole_native actual helper
#axiomsClean matmul_native
#axiomsClean quantize_native
#axiomsClean unpack_native
#axiomsClean whole_native
#axiomsClean whole_fp
end VeriTile.Bench.Repairs.Int4Matmul

namespace VeriTile.Bench.Repairs.Int4Matmul.FloatingSpecification
abbrev matmul_dequantize_int4_s2_native_if_local := @Direct.matmul_native_if_local
abbrev quantize_int4_native_if_local := @Direct.quant_native_if_local
abbrev unpack_int4_native_if_local := @Direct.unpack_native_if_local
#axiomsClean matmul_dequantize_int4_s2_native_if_local
#axiomsClean quantize_int4_native_if_local
#axiomsClean unpack_int4_native_if_local
end VeriTile.Bench.Repairs.Int4Matmul.FloatingSpecification
