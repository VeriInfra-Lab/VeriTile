import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.DirectMemory

namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
open Host Physical Helpers HelperHost
noncomputable section
open Classical
set_option maxHeartbeats 5000000

def checkAllocated (t : Host.Tensor) (dtype : Host.DType) : Except Error Unit :=
  if t.base%dtype.width=0 ∧ t.base+2^33<2^64 then .ok () else .error .pointer
theorem checkAllocated_iff (t : Host.Tensor) (dtype : Host.DType) :
    checkAllocated t dtype=.ok () ↔ t.base%dtype.width=0 ∧ t.base+2^33<2^64 := by
  unfold checkAllocated
  split_ifs <;> simp_all

theorem ordinary_capacity (t : Host.Tensor) (dtype : Host.DType) (rows columns device : Nat)
    (h : t.Ordinary dtype rows columns device) : dtype.width*rows*columns≤t.storageBytes := h.2.2.2.2.2.2.2.2.2.2.2.1

structure DequantClaim (codec : Codec α) (x : PackedArguments) (dtype : Host.DType)
    (a : DequantAllocation x dtype) (ops : DequantOps α) (heap : Heap) (caller : Nat) (result : TensorResult) : Prop where
  identity : result.tensor=a.tensor
  metadata : result.tensor.FreshLayout dtype x.shape.k x.shape.n x.weight.device
  device : result.callerDevice=caller ∧ result.launchDevice=x.weight.device
  bytes : ∀ k n,k<x.shape.k → n<x.shape.n → ∀ i : Fin codec.width,
    result.heap (result.tensor.base+(codec.width*(k*x.shape.n+n)+i.val))=
      codec.encode (dequantValue x.shape ops (load codec (bind (dequantBase x a.tensor) heap)) ⟨k,n⟩) i
  frame : ∀ address,¬(a.tensor.base≤address ∧ address<a.tensor.base+codec.width*x.shape.k*x.shape.n) →
    result.heap address=heap address
  inputs : ∀ t,t∈[x.weight,x.scale,x.zero] → ∀ off,off<t.storageBytes → result.heap (t.base+off)=heap (t.base+off)

theorem dequant_global (codec : Codec α) (x : PackedArguments) (dtype : Host.DType)
    (width : codec.width=dtype.width) (a : DequantAllocation x dtype) (one : α) (ops : DequantOps α)
    (choices : List (Fin 8)) (chosen : Fin 8) (heap : Heap) (caller : Nat) :
    DequantClaim codec x dtype a ops heap caller (globalDequant codec x dtype a one ops choices chosen heap caller) := by
  refine ⟨rfl,a.layout,⟨rfl,rfl⟩,?_,?_,?_⟩
  · exact global_dequant_byte codec x dtype a one ops choices chosen heap caller
  · exact writeSpan_frame _ _ _ heap
  · intro t used off bound
    apply writeSpan_disjoint a.tensor t (a.independent t used)
    · rw [width]
      exact ordinary_capacity _ _ _ _ _ a.layout.1
    · exact bound

def dequantWrapper16 (p : PackedInfo) (device : Option Nat) (a : DequantAllocation (packed p 1) .fp16)
    (I : Native.Instructions) (choices : List (Fin 8)) (chosen : Fin 8) (heap : Heap) (caller : Nat) :
    Except Error TensorResult :=
  match checkDequant p .fp16 device with
  | .error e => .error e
  | .ok _ => match checkAllocated a.tensor .fp16 with
    | .error e => .error e
    | .ok _ => .ok (globalDequant halfCodec (packed p 1) .fp16 a 0x3c00 (Native.dequant16 I) choices chosen heap caller)
def dequantWrapper32 (p : PackedInfo) (device : Option Nat) (a : DequantAllocation (packed p 1) .fp32)
    (I : Native.Instructions) (choices : List (Fin 8)) (chosen : Fin 8) (heap : Heap) (caller : Nat) :
    Except Error TensorResult :=
  match checkDequant p .fp32 device with
  | .error e => .error e
  | .ok _ => match checkAllocated a.tensor .fp32 with
    | .error e => .error e
    | .ok _ => .ok (globalDequant wordCodec (packed p 1) .fp32 a 0x3f800000 (Native.dequant32 I) choices chosen heap caller)

theorem output_disjoint (x : Host.Arguments) (h : Host.Contract x) (a : Host.Allocations x)
    (t : Host.Tensor) (used : t∈[x.input,x.packed.weight,x.packed.scale,x.packed.zero]) :
    Host.Disjoint (x.output a.freshOutput) t := by
  cases hs : x.supplied with
  | none => simpa [Host.Arguments.output,hs] using a.freshDisjoint t used
  | some out => simpa [Host.Arguments.output,hs] using (h.supplied out hs).2 t used

structure S1Claim (x : Host.Arguments) (a : Host.Allocations x) (I : Native.Instructions)
    (plan : Library.Selected x.packed.shape) (heap : Heap) (caller : Nat) (result : TensorResult) : Prop where
  identity : result.tensor=x.output a.freshOutput
  metadata : result.tensor.Ordinary .fp16 x.packed.shape.m x.packed.shape.n x.packed.weight.device
  device : result.callerDevice=caller ∧ result.launchDevice=x.packed.weight.device
  bytes : ∀ r n,r<x.packed.shape.m → n<x.packed.shape.n → ∀ i : Fin 2,
    result.heap (result.tensor.base+(2*(r*x.packed.shape.n+n)+i.val))=
      encode16 (S1.reference x.packed.shape (Native.dequant16 I) (Native.library I) plan
        (load halfCodec (bind (s1Base x a) heap)) r n) i
  frame : ∀ address,¬((x.output a.freshOutput).base≤address ∧
      address<(x.output a.freshOutput).base+2*x.packed.shape.m*x.packed.shape.n) →
    ¬(a.scratch.base≤address ∧ address<a.scratch.base+2*x.packed.shape.k*x.packed.shape.n) →
    result.heap address=heap address
  inputs : ∀ t,t∈[x.input,x.packed.weight,x.packed.scale,x.packed.zero] →
    ∀ off,off<t.storageBytes → result.heap (t.base+off)=heap (t.base+off)

theorem s1_global (x : Host.Arguments) (h : Host.Contract x) (a : Host.Allocations x) (I : Native.Instructions)
    (plan : Library.Selected x.packed.shape) (choices : List (Fin 8)) (chosen : Fin 8) (heap : Heap) (caller : Nat) :
    S1Claim x a I plan heap caller (globalS1 x a I plan choices chosen heap caller) := by
  refine ⟨rfl,return_geometry x h a,⟨rfl,rfl⟩,?_,?_,?_⟩
  · exact global_s1_byte x a I plan choices chosen heap caller
  · intro address ho hs
    change writeSpan _ _ _ _ _=_
    rw [writeSpan_frame _ _ _ _ _ ho,writeSpan_frame _ _ _ _ _ hs]
  · intro t used off bound
    have out := output_disjoint x h a t used
    have scratch := a.scratchDisjoint t (by simp only [List.mem_cons,List.mem_singleton] at used ⊢; tauto)
    change writeSpan _ _ _ _ _=_
    have outcap : 2*x.packed.shape.m*x.packed.shape.n≤(x.output a.freshOutput).storageBytes := ordinary_capacity _ _ _ _ _ (return_geometry x h a)
    rw [writeSpan_disjoint _ _ out _ outcap _ _ off bound]
    have scratchcap : 2*x.packed.shape.k*x.packed.shape.n≤a.scratch.storageBytes := ordinary_capacity _ _ _ _ _ a.scratchLayout.1
    exact writeSpan_disjoint _ _ scratch _ scratchcap _ _ off bound

def checkS1Allocated (p : S1Info) (a : Host.Allocations (arguments p)) : Except Error Unit := do
  checkAllocated ((arguments p).output a.freshOutput) .fp16
  checkAllocated a.scratch .fp16
def s1Wrapper (p : S1Info) (a : Host.Allocations (arguments p)) (I : Native.Instructions)
    (plan : Library.Selected (arguments p).packed.shape) (choices : List (Fin 8)) (chosen : Fin 8)
    (heap : Heap) (caller : Nat) : Except Error TensorResult :=
  match checkS1 p with
  | .error e => .error e
  | .ok _ => match checkS1Allocated p a with
    | .error e => .error e
    | .ok _ => .ok (globalS1 (arguments p) a I plan choices chosen heap caller)

#axiomsClean dequant_global
#axiomsClean s1_global
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
