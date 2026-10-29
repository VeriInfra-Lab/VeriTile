import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectRow
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Copy

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
noncomputable section
set_option maxHeartbeats 2000000

structure HelperRequest where
  weight : TensorView
  ax : Int
  contract : HelperContract weight.info ax
def HelperRequest.column (r : HelperRequest) : Bool := decide (r.ax=0 ∨ r.ax= -2)
def HelperRequest.G (r : HelperRequest) : Nat := if r.column then cols r.weight.info else rows r.weight.info
def HelperRequest.K (r : HelperRequest) : Nat := if r.column then rows r.weight.info else cols r.weight.info
/-- Exact Tensor.contiguous branch on a transpose of a contiguous matrix.
Singleton/empty transposes retain their original storage and inert strides. -/
def HelperRequest.needsCopy (r : HelperRequest) : Bool := decide (r.ax=0 ∧ 1<rows r.weight.info ∧ 1<cols r.weight.info)
def HelperRequest.temporary (r : HelperRequest) (data scales : Nat) : QuantBuffers :=
  ⟨r.weight.info.storage,data,scales,r.weight.info.offset,0,0,
    if r.column then 1 else cols r.weight.info,if r.column then cols r.weight.info else 1,
    r.column,.single,Bool.false⟩
def HelperRequest.destination (r : HelperRequest) (data scales final : Nat) : QuantBuffers :=
  {r.temporary data scales with output:=final,transposeOutput:=if r.needsCopy then Bool.false else r.column}
def ColumnDescriptor (r : HelperRequest) (t : TensorView) : Prop :=
  t.info.metadata=⟨.other "int8",.strided,r.weight.info.metadata.device,[rows r.weight.info,cols r.weight.info],
    Bool.false,Bool.false,t.info.metadata.dataPtr⟩ ∧ t.info.offset=0 ∧
  t.strides=[1,rows r.weight.info] ∧ t.info.transposeContiguous=Bool.true

def helperLogicalIndex (r : HelperRequest) (g k : Nat) : Nat := if r.column then k*cols r.weight.info+g else g*cols r.weight.info+k
/-- Physical element index of the returned view after the intrinsic optional copy. -/
def helperOutputIndex (r : HelperRequest) (g k : Nat) : Nat :=
  if r.needsCopy then g*r.K+k else if r.column then k*r.G+g else g*r.K+k
def HelperViewBinding (r : HelperRequest) (values scales : TensorView) : Prop :=
  (∀ cell : QuantCell r.G r.K,
    logicalAddress values (helperLogicalIndex r cell.1 cell.2)=
      (r.destination values.info.storage scales.info.storage values.info.storage).dataAddress cell) ∧
  (∀ g : Fin r.G,logicalAddress scales g.val=(scales.info.storage,g.val))

structure HelperAllocation (r : HelperRequest) (caller : Memory F H) (live : Nat → Prop) where
  temporary : TensorView
  scales : TensorView
  final : TensorView
  memory : Memory F H
  /-- Torch elementwise allocation may retain inert singleton strides. Its
  exact actual handle is returned on the no-copy branch. -/
  temporaryDescriptor : FreshDescriptor temporary (.other "int8") [rows r.weight.info,cols r.weight.info] temporary.strides r.weight.info.metadata.device
  scaleDescriptor : FreshDescriptor scales (.floating .fp32) [r.G] [1] r.weight.info.metadata.device
  copyDescriptor : r.needsCopy=Bool.true → ColumnDescriptor r final
  noCopyIdentity : r.needsCopy=Bool.false → final=temporary
  fresh : ∀ t,t∈[temporary,scales,final] → ¬live t.info.storage
  separate : temporary.info.storage≠scales.info.storage
  copySeparate : r.needsCopy=Bool.true → temporary.info.storage≠final.info.storage ∧ scales.info.storage≠final.info.storage
  frame : ∀ p,p.1≠temporary.info.storage → p.1≠scales.info.storage → p.1≠final.info.storage → memory p=caller p
  valueIndex : ∀ cell : QuantCell r.G r.K,
    final.physicalIndex (helperLogicalIndex r cell.1 cell.2)=helperOutputIndex r cell.1 cell.2
  scaleIndex : ∀ g : Fin r.G,scales.physicalIndex g.val=g.val
def HelperAllocation.q (alloc : HelperAllocation r caller live) : QuantBuffers := r.temporary alloc.temporary.info.storage alloc.scales.info.storage
def HelperAllocation.dest (alloc : HelperAllocation r caller live) : QuantBuffers := r.destination alloc.temporary.info.storage alloc.scales.info.storage alloc.final.info.storage
theorem HelperAllocation.binding (alloc : HelperAllocation r caller live) : HelperViewBinding r alloc.final alloc.scales := by
  constructor
  · intro cell
    unfold logicalAddress
    rw [alloc.valueIndex cell]
    cases copy : r.needsCopy <;> cases column : r.column <;>
      simp [helperOutputIndex,HelperRequest.destination,HelperRequest.temporary,QuantBuffers.dataAddress,copy,column]
  · intro g
    unfold logicalAddress
    rw [alloc.scaleIndex g]

theorem HelperAllocation.inputSeparate (r : HelperRequest) (caller : Memory F H) (live : Nat → Prop)
    (input : live r.weight.info.storage) (alloc : HelperAllocation r caller live) : alloc.q.Separate := by
  refine ⟨?_,?_,alloc.separate⟩
  · intro h;change r.weight.info.storage=alloc.temporary.info.storage at h;rw [h] at input;exact alloc.fresh _ (by simp) input
  · intro h;change r.weight.info.storage=alloc.scales.info.storage at h;rw [h] at input;exact alloc.fresh _ (by simp) input
theorem HelperAllocation.inputFrame (r : HelperRequest) (caller : Memory F H) (live : Nat → Prop)
    (input : live r.weight.info.storage) (alloc : HelperAllocation r caller live) : QuantInputsAgree alloc.q alloc.memory caller := by
  intro p hp
  have outside : ∀ t,t∈[alloc.temporary,alloc.scales,alloc.final] → p.1≠t.info.storage := by
    intro t ht eq
    have livep : live p.1 := by simpa only [hp] using input
    rw [eq] at livep;exact alloc.fresh t ht livep
  exact alloc.frame p (outside _ (by simp)) (outside _ (by simp)) (outside _ (by simp))
theorem HelperAllocation.noCopy (alloc : HelperAllocation r caller live) (h : r.needsCopy=Bool.false) : alloc.dest=alloc.q := by
  simp [HelperAllocation.dest,HelperAllocation.q,HelperRequest.destination,HelperRequest.temporary,h,alloc.noCopyIdentity h]

def helperEffect (a : Ops I F H) (r : HelperRequest) (q dest : QuantBuffers)
    (chosen : QuantInvocation r.G r.K) (m : Memory F H) :=
  if r.needsCopy then copyRun a q dest r.G r.K (quantRun a q chosen m) else quantRun a q chosen m
def helperEffectCode (r : HelperRequest) (q dest : QuantBuffers) (chosen : QuantInvocation r.G r.K) : Code I F H Unit := do
  let _ ← quantGridCode q chosen.plan chosen.groups
  if r.needsCopy then copyCode q.dataAddress dest.dataAddress (quantCells r.G r.K) else pure ()
theorem helperEffectCode_eval (a : Ops I F H) (r : HelperRequest) (q dest : QuantBuffers)
    (chosen : QuantInvocation r.G r.K) (m : Memory F H) :
    (helperEffectCode r q dest chosen).eval a m=((),helperEffect a r q dest chosen m) := by
  unfold helperEffectCode helperEffect
  cases h : r.needsCopy <;> simp [h,bind,eval_bind,pure,eval_pure,copyRun,quantRun]

structure HelperResult (F H : Type) where
  values : TensorView
  scales : TensorView
  third : Option Unit
  memory : Memory F H
  currentDevice : Nat
def runHelper (r : HelperRequest) (a : Ops I F H) (callerDevice : Nat) (chosen : QuantInvocation r.G r.K)
    (caller : Memory F H) (live : Nat → Prop) (alloc : HelperAllocation r caller live) : Except InputError (HelperResult F H) := do
  checkHelper (.tensor r.weight.info) (.integer r.ax)
  return ⟨alloc.final,alloc.scales,none,helperEffect a r alloc.q alloc.dest chosen alloc.memory,callerDevice⟩

theorem helper_values (r : HelperRequest) (a : Ops I F H) (chosen : QuantInvocation r.G r.K)
    (caller : Memory F H) (live : Nat → Prop) (input : live r.weight.info.storage) (alloc : HelperAllocation r caller live)
    (cell : QuantCell r.G r.K) :
    helperEffect a r alloc.q alloc.dest chosen alloc.memory (alloc.dest.dataAddress cell)=.inl (quantValue a alloc.q chosen.plan caller cell) := by
  have sep:=alloc.inputSeparate r caller live input
  have same:=alloc.inputFrame r caller live input
  unfold helperEffect
  cases h : r.needsCopy with
  | «true»=>
    simp only [h,ite_true]
    rw [copyRun_at a alloc.q alloc.dest (alloc.copySeparate h).1]
    have value:=quantRun_at a alloc.q sep chosen alloc.memory (.inl cell)
    change quantRun a alloc.q chosen alloc.memory (alloc.q.dataAddress cell)=.inl _ at value
    simp only [read8,Int8DequantMatmul.read8,value,quantValue_current a alloc.q _ _ _ same]
  | «false»=>
    simp only [h,Bool.false_eq_true,ite_false,alloc.noCopy h]
    have value:=quantRun_at a alloc.q sep chosen alloc.memory (.inl cell)
    change quantRun a alloc.q chosen alloc.memory (alloc.q.dataAddress cell)=.inl _ at value
    exact value.trans (congrArg Sum.inl (quantValue_current a alloc.q _ _ _ same cell))
theorem helper_scales (r : HelperRequest) (a : Ops I F H) (chosen : QuantInvocation r.G r.K)
    (caller : Memory F H) (live : Nat → Prop) (input : live r.weight.info.storage) (alloc : HelperAllocation r caller live)
    (g : Fin r.G) :
    helperEffect a r alloc.q alloc.dest chosen alloc.memory (alloc.q.scaleAddress g)=.inr (.inl (quantScale a alloc.q chosen.plan caller g)) := by
  have sep:=alloc.inputSeparate r caller live input
  have same:=alloc.inputFrame r caller live input
  have scale:=quantRun_at a alloc.q sep chosen alloc.memory (.inr g)
  change quantRun a alloc.q chosen alloc.memory (alloc.q.scaleAddress g)=.inr (.inl (quantScale a alloc.q chosen.plan alloc.memory g)) at scale
  rw [quantScale_current a alloc.q _ _ _ same] at scale
  unfold helperEffect
  cases h : r.needsCopy with
  | «true»=>
    simp only [h,ite_true]
    rw [copyRun_frame a alloc.q alloc.dest (alloc.copySeparate h).1]
    · exact scale
    · intro x eq;exact (alloc.copySeparate h).2 (congrArg Prod.fst eq)
  | «false»=>simpa [h] using scale
theorem helper_frame (r : HelperRequest) (a : Ops I F H) (chosen : QuantInvocation r.G r.K)
    (caller : Memory F H) (live : Nat → Prop) (input : live r.weight.info.storage) (alloc : HelperAllocation r caller live)
    (p : Address) (owned : live p.1) : helperEffect a r alloc.q alloc.dest chosen alloc.memory p=caller p := by
  have outside : ∀ t,t∈[alloc.temporary,alloc.scales,alloc.final] → p.1≠t.info.storage := by
    intro t ht eq;rw [eq] at owned;exact alloc.fresh t ht owned
  have quant : quantRun a alloc.q chosen alloc.memory p=alloc.memory p := by
    apply quantRun_frame a alloc.q (alloc.inputSeparate r caller live input)
    intro idx eq
    have h:=congrArg Prod.fst eq
    cases idx with
    | inl x=>exact outside _ (by simp) h
    | inr g=>exact outside _ (by simp) h
  unfold helperEffect
  cases h : r.needsCopy with
  | «true»=>
    simp only [h,ite_true]
    rw [copyRun_frame a alloc.q alloc.dest (alloc.copySeparate h).1]
    · rw [quant];exact alloc.frame p (outside _ (by simp)) (outside _ (by simp)) (outside _ (by simp))
    · intro x eq;exact outside _ (by simp) (congrArg Prod.fst eq)
  | «false»=>simp only [h,Bool.false_eq_true,ite_false,quant];exact alloc.frame p (outside _ (by simp)) (outside _ (by simp)) (outside _ (by simp))

theorem helper_complete (r : HelperRequest) (a : Ops I F H) (callerDevice : Nat) (chosen : QuantInvocation r.G r.K)
    (caller : Memory F H) (live : Nat → Prop) (input : live r.weight.info.storage) (alloc : HelperAllocation r caller live) :
    ∃ result,runHelper r a callerDevice chosen caller live alloc=.ok result ∧
      result.values=alloc.final ∧ result.scales=alloc.scales ∧ result.third=none ∧ result.currentDevice=callerDevice ∧
      (r.needsCopy=Bool.true → ColumnDescriptor r result.values) ∧
      (r.needsCopy=Bool.false → result.values=alloc.temporary) ∧
      FreshDescriptor result.scales (.floating .fp32) [r.G] [1] r.weight.info.metadata.device ∧
      HelperViewBinding r result.values result.scales ∧
      (∀ cell : QuantCell r.G r.K,result.memory (alloc.dest.dataAddress cell)=.inl (quantValue a alloc.q chosen.plan caller cell)) ∧
      (∀ g : Fin r.G,result.memory (alloc.q.scaleAddress g)=.inr (.inl (quantScale a alloc.q chosen.plan caller g))) ∧
      (∀ p,live p.1 → result.memory p=caller p) := by
  refine ⟨⟨alloc.final,alloc.scales,none,helperEffect a r alloc.q alloc.dest chosen alloc.memory,callerDevice⟩,
    ?_,rfl,rfl,rfl,rfl,alloc.copyDescriptor,alloc.noCopyIdentity,alloc.scaleDescriptor,alloc.binding,?_,?_,?_⟩
  · simp [runHelper,(checkHelper_iff (.tensor r.weight.info) (.integer r.ax)).mpr r.contract,Except.bind,Except.pure,bind,pure]
  · exact helper_values r a chosen caller live input alloc
  · exact helper_scales r a chosen caller live input alloc
  · exact helper_frame r a chosen caller live input alloc

def helperOriginal (a : Ops I F H) (r : HelperRequest) (m : Memory F H) (g k : Nat) : F :=
  read32 a.toOps m (logicalAddress r.weight (helperLogicalIndex r g k))
theorem helper_input_address (base : Nat → Nat) (r : HelperRequest) (laws : TensorLaws base r.weight)
    (data scales g k : Nat) (hg : g<r.G) (hk : k<r.K) :
    (r.temporary data scales).inputAddress g k=logicalAddress r.weight (helperLogicalIndex r g k) := by
  have bound : helperLogicalIndex r g k<r.weight.info.metadata.numel := by
    rw [matrix_numel _ r.contract.input.rank]
    cases h : r.column <;> simp only [HelperRequest.G,HelperRequest.K,h,Bool.false_eq_true,ite_false,ite_true] at hg hk <;>
      simp only [helperLogicalIndex,h,Bool.false_eq_true,ite_false,ite_true] <;> nlinarith
  unfold logicalAddress
  rw [laws.contiguous (row_contiguous r.contract.input) _ bound]
  cases h : r.column <;> simp [HelperRequest.temporary,QuantBuffers.inputAddress,helperLogicalIndex,h,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
def helperScaleTarget (r : HelperRequest) (m : Memory ℝ ℝ) (g : Nat) := quantScaleTarget (helperOriginal realOps r m g) r.K
def helperValueTarget (r : HelperRequest) (m : Memory ℝ ℝ) (g k : Nat) := quantValueTarget (helperOriginal realOps r m g) r.K k
theorem helper_scale_original (base : Nat → Nat) (r : HelperRequest) (laws : TensorLaws base r.weight)
    (data scales : Nat) (plan : Fin r.G → ReductionPlan r.K) (m : Memory ℝ ℝ) (g : Fin r.G) :
    quantScale realOps (r.temporary data scales) plan m g=helperScaleTarget r m g := by
  rw [quantScale_real]
  unfold helperScaleTarget quantScaleTarget
  congr 1
  apply maximumAbs_congr
  intro k hk;rw [helper_input_address base r laws data scales g k g.isLt hk];rfl
theorem helper_value_original (base : Nat → Nat) (r : HelperRequest) (laws : TensorLaws base r.weight)
    (data scales : Nat) (plan : Fin r.G → ReductionPlan r.K) (m : Memory ℝ ℝ) (cell : QuantCell r.G r.K) :
    quantValue realOps (r.temporary data scales) plan m cell=helperValueTarget r m cell.1 cell.2 := by
  unfold quantValue
  rw [helper_scale_original base r laws,helper_input_address base r laws data scales cell.1 cell.2 cell.1.isLt cell.2.isLt]
  rfl

#axiomsClean helperEffectCode_eval
#axiomsClean helper_complete
#axiomsClean helper_input_address
#axiomsClean helper_scale_original
#axiomsClean helper_value_original
#specNonCircular helperScaleTarget avoiding [helperEffect,quantScale,Reduction.eval]
#specNonCircular helperValueTarget avoiding [helperEffect,quantValue,Reduction.eval]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
