import bench.repairs.tritonbench_g.int4_matmul.Proofs.DirectHelpersData

namespace VeriTile.Bench.Repairs.Int4Matmul.Direct
open Host Helpers HelperHost HelperBytes Physical
noncomputable section
set_option maxHeartbeats 2500000

instance (w : Word) : Decidable (finite32 w) := inferInstanceAs (Decidable (_≠_))
instance (w : Word) : Decidable (positive32 w) := inferInstanceAs (Decidable (_∧_∧_))
def allMatrix (rows cols : Nat) (test : Nat → Nat → Bool) : Bool :=
  (List.range rows).all (fun row=>(List.range cols).all (fun col=>test row col))

def checkQuantWords (r : QuantRequest) (p : HelperNative.Instructions) (order : ReductionOrder r.shape.group) (caller : ByteHeap) : Except InputError Unit :=
  if !(allMatrix r.shape.k r.shape.n (fun k n=>decide (finite32 ((load (quantBytesInput r caller)).values .weight (k*r.shape.n+n))))) then .error .finite
  else if !(allMatrix (r.shape.k/r.shape.group) r.shape.n (fun g n=>decide (positive32 (scale r.shape (HelperNative.arithmetic p) order (load (quantBytesInput r caller)) g n)))) then .error .scale
  else .ok ()
structure QuantWordDomain (r : QuantRequest) (p : HelperNative.Instructions) (order : ReductionOrder r.shape.group) (caller : ByteHeap) : Prop where
  finite : ∀ k,k<r.shape.k → ∀ n,n<r.shape.n → finite32 ((load (quantBytesInput r caller)).values .weight (k*r.shape.n+n))
  scale : ∀ g,g<r.shape.k/r.shape.group → ∀ n,n<r.shape.n → positive32 (Helpers.scale r.shape (HelperNative.arithmetic p) order (load (quantBytesInput r caller)) g n)
theorem checkQuantWords_iff (r : QuantRequest) (p : HelperNative.Instructions) (order : ReductionOrder r.shape.group) (caller : ByteHeap) :
    checkQuantWords r p order caller=.ok () ↔ QuantWordDomain r p order caller := by
  have domain : QuantWordDomain r p order caller ↔
      (∀ k,k<r.shape.k → ∀ n,n<r.shape.n → finite32 ((load (quantBytesInput r caller)).values .weight (k*r.shape.n+n))) ∧
      (∀ g,g<r.shape.k/r.shape.group → ∀ n,n<r.shape.n → positive32 (scale r.shape (HelperNative.arithmetic p) order (load (quantBytesInput r caller)) g n)) := by
    constructor
    · intro h;exact ⟨h.finite,h.scale⟩
    · rintro ⟨a,b⟩;exact ⟨a,b⟩
  rw [domain]
  simp [checkQuantWords,allMatrix,ite_eq_iff]

structure QuantNativeResult where
  weights : TensorInfo
  scales : TensorInfo
  zeros : TensorInfo
  group : Int
  memory : ByteHeap
  currentDevice : Nat

def quantMerge (alloc : QuantAllocation r live) (caller : ByteHeap) (result : QuantizedBytes) : ByteHeap :=
  fun storage i=>if storage=alloc.weights.descriptor.allocation then result.weights i
    else if storage=alloc.scales.descriptor.allocation then result.scales i
    else if storage=alloc.zeros.descriptor.allocation then result.zeros i else caller storage i
def quantScratch (alloc : QuantAllocation r live) (caller : ByteHeap) : Fin 3 → Nat → Byte :=
  fun slot=>if slot.val=0 then caller alloc.weights.descriptor.allocation
    else if slot.val=1 then caller alloc.scales.descriptor.allocation else caller alloc.zeros.descriptor.allocation

def runQuantNative (r : QuantRequest) (p : HelperNative.Instructions) (call : HelperSource r) (callerDevice : Nat)
    (caller : ByteHeap) (live : Nat → Prop) (alloc : QuantAllocation r live) : Except InputError QuantNativeResult := do
  checkQuantMetadata r.weight r.group r.rank
  checkQuantWords r p call.order caller
  let result:=quantizeBytes r.shape (HelperNative.arithmetic p) call.order (quantBytesInput r caller) (quantScratch alloc caller)
  return ⟨alloc.weights,alloc.scales,alloc.zeros,r.group,quantMerge alloc caller result,callerDevice⟩

def originalWeightTarget (r : QuantRequest) (p : HelperNative.Instructions) (order : ReductionOrder r.shape.group)
    (caller : ByteHeap) (k n : Nat) := weightSpec r.shape (HelperNative.arithmetic p) order (originalQuantState r (heapWords caller)) k n
def originalZeroTarget (r : QuantRequest) (p : HelperNative.Instructions) (order : ReductionOrder r.shape.group)
    (caller : ByteHeap) (g n : Nat) := zeroSpec r.shape (HelperNative.arithmetic p) order (originalQuantState r (heapWords caller)) g n
def originalScaleTarget (r : QuantRequest) (p : HelperNative.Instructions) (order : ReductionOrder r.shape.group)
    (caller : ByteHeap) (g n : Nat) := scale r.shape (HelperNative.arithmetic p) order (originalQuantState r (heapWords caller)) g n

theorem quant_native_if_local (base : Nat → Nat) (r : QuantRequest) (inputLaws : TensorLaws base r.weight)
    (p reference : HelperNative.Instructions) (numeric : HelperNative.Correspondence p reference)
    (call : HelperSource r) (callerDevice : Nat) (caller : ByteHeap) (live : Nat → Prop)
    (input : live r.weight.descriptor.allocation)
    (admitted : QuantWordDomain r p call.order caller) (alloc : QuantAllocation r live)
    (outputLaws : ∀ t,t∈[alloc.weights,alloc.scales,alloc.zeros] → TensorLaws base t) :
    ∃ result,runQuantNative r p call callerDevice caller live alloc=.ok result ∧
      result.weights=alloc.weights ∧ result.scales=alloc.scales ∧ result.zeros=alloc.zeros ∧ result.group=r.group ∧
      QuantTensor result.weights .int32 (r.shape.k/8) r.shape.n r.weight.descriptor.device ∧
      QuantTensor result.scales .fp32 (r.shape.k/r.shape.group) r.shape.n r.weight.descriptor.device ∧
      QuantTensor result.zeros .int32 (r.shape.k/r.shape.group) (r.shape.n/8) r.weight.descriptor.device ∧
      result.currentDevice=callerDevice ∧ PointerBinding base r.weight ∧
      (∀ t,t∈[result.weights,result.scales,result.zeros] → PointerBinding base t) ∧
      (∀ k n,k<r.shape.k/8 → n<r.shape.n → ∀ i : Fin 4,
        result.memory result.weights.descriptor.allocation (4*logicalIndex result.weights k n+i.val)=encode (originalWeightTarget r reference call.order caller k n) i) ∧
      (∀ g n,g<r.shape.k/r.shape.group → n<r.shape.n → ∀ i : Fin 4,
        result.memory result.scales.descriptor.allocation (4*logicalIndex result.scales g n+i.val)=encode (originalScaleTarget r reference call.order caller g n) i) ∧
      (∀ g n,g<r.shape.k/r.shape.group → n<r.shape.n/8 → ∀ i : Fin 4,
        result.memory result.zeros.descriptor.allocation (4*logicalIndex result.zeros g n+i.val)=encode (originalZeroTarget r reference call.order caller g n) i) ∧
      (∀ storage i,live storage → result.memory storage i=caller storage i) ∧ call.parsedProduction=SourceBinding.productionAST := by
  let result:=quantizeBytes r.shape (HelperNative.arithmetic p) call.order (quantBytesInput r caller) (quantScratch alloc caller)
  refine ⟨⟨alloc.weights,alloc.scales,alloc.zeros,r.group,quantMerge alloc caller result,callerDevice⟩,
    ?_,rfl,rfl,rfl,rfl,alloc.weightsDescriptor,alloc.scalesDescriptor,alloc.zerosDescriptor,rfl,pointer_binding base r.weight inputLaws,
    (fun t ht=>pointer_binding base t (outputLaws t ht)),?_,?_,?_,?_,call.sourceIdentity⟩
  · simp [runQuantNative,(checkQuantMetadata_iff _ _ _).mpr r.contract,(checkQuantWords_iff _ _ _ _).mpr admitted,Except.bind,Except.pure,bind,pure,result]
  · intro k n hk hn i
    change quantMerge alloc caller result alloc.weights.descriptor.allocation (4*logicalIndex alloc.weights k n+i.val)=_
    simp only [quantMerge,ite_true,quantTensor_index _ _ _ _ _ _ _ alloc.weightsDescriptor hk hn]
    dsimp only [result]
    rw [quantize_weight_bytes _ r.valid _ _ _ _ k n hk hn i,quantBytesInput_load,quantState_original,HelperNative.correspondence_eq p reference numeric]
    rfl
  · intro g n hg hn i
    change quantMerge alloc caller result alloc.scales.descriptor.allocation (4*logicalIndex alloc.scales g n+i.val)=_
    simp only [quantMerge,Ne.symm alloc.separate.1,ite_false,ite_true,quantTensor_index _ _ _ _ _ _ _ alloc.scalesDescriptor hg hn]
    dsimp only [result]
    rw [quantize_scale_bytes _ r.valid _ _ _ _ g n hg hn i,quantBytesInput_load,quantState_original,HelperNative.correspondence_eq p reference numeric]
    rfl
  · intro g n hg hn i
    change quantMerge alloc caller result alloc.zeros.descriptor.allocation (4*logicalIndex alloc.zeros g n+i.val)=_
    simp only [quantMerge,Ne.symm alloc.separate.2.1,Ne.symm alloc.separate.2.2,ite_false,ite_true,quantTensor_index _ _ _ _ _ _ _ alloc.zerosDescriptor hg hn]
    dsimp only [result]
    rw [quantize_zero_bytes _ r.valid _ _ _ _ g n hg hn i,quantBytesInput_load,quantState_original,HelperNative.correspondence_eq p reference numeric]
    rfl
  · intro storage i owned
    have outside (t : TensorInfo) (ht : t∈[alloc.weights,alloc.scales,alloc.zeros]) : storage≠t.descriptor.allocation:=by
      intro he;rw [he] at owned;exact alloc.fresh t ht owned
    simp only [quantMerge,outside alloc.weights (by simp),outside alloc.scales (by simp),outside alloc.zeros (by simp),ite_false]

#axiomsClean checkQuantWords_iff
#axiomsClean quant_native_if_local
#specNonCircular originalWeightTarget avoiding [runQuantNative,quantizeBytes,quantize,packLoop]
#specNonCircular originalZeroTarget avoiding [runQuantNative,quantizeBytes,quantize,packLoop]
#specNonCircular originalScaleTarget avoiding [runQuantNative,quantizeBytes,quantize,packLoop]
end
end VeriTile.Bench.Repairs.Int4Matmul.Direct
