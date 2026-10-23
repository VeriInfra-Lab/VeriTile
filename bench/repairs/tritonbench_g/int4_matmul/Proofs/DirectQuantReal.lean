import bench.repairs.tritonbench_g.int4_matmul.Proofs.DirectQuantNative

namespace VeriTile.Bench.Repairs.Int4Matmul.Direct
open Host Helpers HelperHost HelperBytes Physical
noncomputable section
set_option maxHeartbeats 2500000

/-- The exact mathematical finite-range interpretation uses arbitrary Reals;
there is no encoding of those values into binary32 words. -/
def realFinite32 (x : ℝ) : Prop := |x|≤((2:ℝ)^24-1)*(2:ℝ)^104
local instance (x : ℝ) : Decidable (realFinite32 x) := Classical.propDecidable _
local instance (x : ℝ) : Decidable (0<x ∧ realFinite32 x) := Classical.propDecidable _

def checkQuantReals (r : QuantRequest) (order : ReductionOrder r.shape.group) (caller : HeapState ℝ) : Except InputError Unit :=
  if !(allMatrix r.shape.k r.shape.n (fun k n=>decide (realFinite32 ((quantState r caller).values .weight (k*r.shape.n+n))))) then .error .finite
  else if !(allMatrix (r.shape.k/r.shape.group) r.shape.n (fun g n=>decide
    (0<scale r.shape HelperReal.arithmetic order (quantState r caller) g n ∧ realFinite32 (scale r.shape HelperReal.arithmetic order (quantState r caller) g n)))) then .error .scale
  else .ok ()
structure QuantRealDomain (r : QuantRequest) (order : ReductionOrder r.shape.group) (caller : HeapState ℝ) : Prop where
  finite : ∀ k,k<r.shape.k → ∀ n,n<r.shape.n → realFinite32 ((quantState r caller).values .weight (k*r.shape.n+n))
  scale : ∀ g,g<r.shape.k/r.shape.group → ∀ n,n<r.shape.n →
    0<Helpers.scale r.shape HelperReal.arithmetic order (quantState r caller) g n ∧ realFinite32 (Helpers.scale r.shape HelperReal.arithmetic order (quantState r caller) g n)
theorem checkQuantReals_iff (r : QuantRequest) (order : ReductionOrder r.shape.group) (caller : HeapState ℝ) :
    checkQuantReals r order caller=.ok () ↔ QuantRealDomain r order caller := by
  have domain : QuantRealDomain r order caller ↔
      (∀ k,k<r.shape.k → ∀ n,n<r.shape.n → realFinite32 ((quantState r caller).values .weight (k*r.shape.n+n))) ∧
      (∀ g,g<r.shape.k/r.shape.group → ∀ n,n<r.shape.n →
        0<scale r.shape HelperReal.arithmetic order (quantState r caller) g n ∧ realFinite32 (scale r.shape HelperReal.arithmetic order (quantState r caller) g n)) := by
    constructor
    · intro h;exact ⟨h.finite,h.scale⟩
    · rintro ⟨a,b⟩;exact ⟨a,b⟩
  rw [domain]
  simp [checkQuantReals,allMatrix,ite_eq_iff]

structure QuantRealResult where
  weights : TensorInfo
  scales : TensorInfo
  zeros : TensorInfo
  group : Int
  memory : HeapState ℝ
  currentDevice : Nat

def quantRealMerge (r : QuantRequest) (alloc : QuantAllocation r live) (caller : HeapState ℝ) (result : Quantized ℝ) : HeapState ℝ :=
  ⟨(fun storage i=>if storage=alloc.scales.descriptor.allocation ∧ i<(r.shape.k/r.shape.group)*r.shape.n then result.scales i else caller.values storage i),
   (fun storage i=>if storage=alloc.weights.descriptor.allocation ∧ i<(r.shape.k/8)*r.shape.n then result.weights i
    else if storage=alloc.zeros.descriptor.allocation ∧ i<(r.shape.k/r.shape.group)*(r.shape.n/8) then result.zeros i else caller.packed storage i)⟩
def runQuantReal (r : QuantRequest) (order : ReductionOrder r.shape.group) (callerDevice : Nat)
    (caller : HeapState ℝ) (live : Nat → Prop) (alloc : QuantAllocation r live) : Except InputError QuantRealResult := do
  checkQuantMetadata r.weight r.group r.rank
  checkQuantReals r order caller
  let result:=quantize r.shape HelperReal.arithmetic order (quantState r caller)
  return ⟨alloc.weights,alloc.scales,alloc.zeros,r.group,quantRealMerge r alloc caller result,callerDevice⟩

def realWeightTarget (r : QuantRequest) (order : ReductionOrder r.shape.group) (caller : HeapState ℝ) (k n : Nat) :=
  weightSpec r.shape HelperReal.arithmetic order (originalQuantState r caller) k n
def realZeroTarget (r : QuantRequest) (order : ReductionOrder r.shape.group) (caller : HeapState ℝ) (g n : Nat) :=
  zeroSpec r.shape HelperReal.arithmetic order (originalQuantState r caller) g n
def realScaleTarget (r : QuantRequest) (order : ReductionOrder r.shape.group) (caller : HeapState ℝ) (g n : Nat) :=
  scale r.shape HelperReal.arithmetic order (originalQuantState r caller) g n

theorem quant_real_complete (r : QuantRequest) (order : ReductionOrder r.shape.group) (callerDevice : Nat)
    (caller : HeapState ℝ) (live : Nat → Prop) (input : live r.weight.descriptor.allocation)
    (admitted : QuantRealDomain r order caller) (alloc : QuantAllocation r live) :
    ∃ result,runQuantReal r order callerDevice caller live alloc=.ok result ∧
      result.weights=alloc.weights ∧ result.scales=alloc.scales ∧ result.zeros=alloc.zeros ∧ result.group=r.group ∧
      QuantTensor result.weights .int32 (r.shape.k/8) r.shape.n r.weight.descriptor.device ∧
      QuantTensor result.scales .fp32 (r.shape.k/r.shape.group) r.shape.n r.weight.descriptor.device ∧
      QuantTensor result.zeros .int32 (r.shape.k/r.shape.group) (r.shape.n/8) r.weight.descriptor.device ∧
      result.currentDevice=callerDevice ∧
      (∀ k n,k<r.shape.k/8 → n<r.shape.n → result.memory.packed result.weights.descriptor.allocation
        (logicalIndex result.weights k n)=realWeightTarget r order caller k n) ∧
      (∀ g n,g<r.shape.k/r.shape.group → n<r.shape.n → result.memory.values result.scales.descriptor.allocation
        (logicalIndex result.scales g n)=realScaleTarget r order caller g n) ∧
      (∀ g n,g<r.shape.k/r.shape.group → n<r.shape.n/8 → result.memory.packed result.zeros.descriptor.allocation
        (logicalIndex result.zeros g n)=realZeroTarget r order caller g n) ∧
      (∀ storage i,live storage → result.memory.values storage i=caller.values storage i ∧ result.memory.packed storage i=caller.packed storage i) ∧
      (∀ g n,∀ hi lo,HelperReal.GroupRange (fun i : Fin r.shape.group=>Helpers.input r.shape (originalQuantState r caller) g n i) hi lo →
        realScaleTarget r order caller g n=(hi-lo)/15) := by
  let q:=quantize r.shape HelperReal.arithmetic order (quantState r caller)
  refine ⟨⟨alloc.weights,alloc.scales,alloc.zeros,r.group,quantRealMerge r alloc caller q,callerDevice⟩,
    ?_,rfl,rfl,rfl,rfl,alloc.weightsDescriptor,alloc.scalesDescriptor,alloc.zerosDescriptor,rfl,?_,?_,?_,?_,?_⟩
  · simp [runQuantReal,(checkQuantMetadata_iff _ _ _).mpr r.contract,(checkQuantReals_iff _ _ _).mpr admitted,Except.bind,Except.pure,bind,pure,q]
  · intro k n hk hn
    change (quantRealMerge r alloc caller q).packed alloc.weights.descriptor.allocation (logicalIndex alloc.weights k n)=_
    simp only [quantTensor_index _ _ _ _ _ _ _ alloc.weightsDescriptor hk hn,quantRealMerge,rowMajor_bound _ _ _ _ hk hn,true_and,ite_true]
    dsimp only [q]
    rw [quantize_result _ r.valid _ _ _ k n hk hn,quantState_original]
    rfl
  · intro g n hg hn
    change (quantRealMerge r alloc caller q).values alloc.scales.descriptor.allocation (logicalIndex alloc.scales g n)=_
    simp only [quantTensor_index _ _ _ _ _ _ _ alloc.scalesDescriptor hg hn,quantRealMerge,rowMajor_bound _ _ _ _ hg hn,true_and,ite_true]
    dsimp only [q]
    rw [quantize_scales _ r.valid _ _ _ g n hn,quantState_original]
    rfl
  · intro g n hg hn
    change (quantRealMerge r alloc caller q).packed alloc.zeros.descriptor.allocation (logicalIndex alloc.zeros g n)=_
    simp only [quantTensor_index _ _ _ _ _ _ _ alloc.zerosDescriptor hg hn,quantRealMerge,Ne.symm alloc.separate.2.1,false_and,
      rowMajor_bound _ _ _ _ hg hn,true_and,ite_false,ite_true]
    dsimp only [q]
    rw [quantize_zeros _ r.valid _ _ _ g n hn,quantState_original]
    rfl
  · intro storage i owned
    have outside (t : TensorInfo) (ht : t∈[alloc.weights,alloc.scales,alloc.zeros]) : storage≠t.descriptor.allocation:=by
      intro he;rw [he] at owned;exact alloc.fresh t ht owned
    simp only [quantRealMerge,outside alloc.weights (by simp),outside alloc.scales (by simp),outside alloc.zeros (by simp),false_and,ite_false,and_self]
  · intro g n hi lo range
    exact HelperReal.scale_independent r.shape order (originalQuantState r caller) g n hi lo range

#axiomsClean checkQuantReals_iff
#axiomsClean quant_real_complete
#specNonCircular realWeightTarget avoiding [runQuantReal,quantRealMerge,quantize,packLoop]
#specNonCircular realZeroTarget avoiding [runQuantReal,quantRealMerge,quantize,packLoop]
#specNonCircular realScaleTarget avoiding [runQuantReal,quantRealMerge,quantize,packLoop]
end
end VeriTile.Bench.Repairs.Int4Matmul.Direct
