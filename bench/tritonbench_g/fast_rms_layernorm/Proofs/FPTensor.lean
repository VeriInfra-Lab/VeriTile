import bench.tritonbench_g.fast_rms_layernorm.Proofs.KernelFrame

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPTensor
open VeriTile.Triton FP.Structural KernelTensor KernelStorage
open FPGridForward (Heap entry)
noncomputable section
variable {α : Type} [Inhabited α]

/-- Independent source-order references over logical tensor elements. The
padding and FP32 cast/load edges are in FPReference, not in these tensors. -/
def row (h : Call) (logical : RegionName → Nat → α) (role : RegionName) (i : Fin h.rows) : Fin h.block → α :=
  fun j => logical role (i.val*h.cols+j.val)
def weight (h : Call) (logical : RegionName → Nat → α) : Fin h.block → α := fun j => logical "w" j.val

def forwardReference (M : Algebra α) (h : Call) (logical : RegionName → Nat → α) (i : Fin h.rows) (j : Fin h.cols) :=
  FPReference.output M h (row h logical "x" i) (weight h logical) ⟨j.val,j.isLt.trans_le h.covered⟩
def savedReference (M : Algebra α) (h : Call) (logical : RegionName → Nat → α) (i : Fin h.rows) :=
  FPReference.savedInv M h (row h logical "x" i)
def inputReference (M : Algebra α) (h : Call) (logical : RegionName → Nat → α) (i : Fin h.rows) (j : Fin h.cols) :=
  FPBackwardReference.output M h (row h logical "x" i) (weight h logical) (row h logical "dx" i)
    (logical "r" i.val) ⟨j.val,j.isLt.trans_le h.covered⟩
def weightReference (M : Algebra α) (h : Call) (logical : RegionName → Nat → α) (j : Fin h.cols) :=
  FPWeightReference.output M h.gemma (fun i => logical "x" (i*h.cols+j.val))
    (fun i => logical "dx" (i*h.cols+j.val)) (logical "r") h.rows

theorem padded_agrees (M : Algebra α) (h : Call) (a b : Fin h.block → α)
    (agree : ∀ j, j.val<h.cols → a j=b j) : FPReference.padded M h a=FPReference.padded M h b := by
  funext j
  by_cases hi : j.val<h.cols
  · simp [FPReference.padded,hi,agree j hi]
  · simp [FPReference.padded,hi]

/-- Each launch has its own algebra: dW can require split mul/add while dX
retains the compiler-selected contraction structure. -/
def sequence (W X : Algebra α) (h : Call) (initial : FPGridForward.Heap α) : Option (FPGridForward.Heap α) :=
  (FPGridWeight.run W h initial).bind (FPGridBackward.run X h)

theorem sequence_complete (W X : Algebra α) (h : Call) (heap : FPGridForward.Heap α) :
    ∃ final,sequence W X h heap=some final ∧
      (∀ i : Fin h.rows,∀ j,final (FPGridBackward.address h i j)=FPGridBackward.expected X h heap i j) ∧
      (∀ pid : Fin (Geometry.featureCount h),∀ j,final (FPGridWeight.address h pid j)=FPGridWeight.expected W h heap pid j) ∧
      Frame h .backward heap final := by
  obtain ⟨weights,hw,wo,wf⟩:=FPGridWeight.complete W h heap
  obtain ⟨final,hx,xo,xf⟩:=FPGridBackward.complete X h weights
  refine ⟨final,by simp [sequence,hw,hx],?_,?_,backward_frame h heap final ?_⟩
  · intro i j
    rw [xo]
    apply FPGridBackward.expected_stable X h i weights heap
    intro addr read
    apply wf
    intro pid
    rintro ⟨lane,eq⟩
    subst addr
    simp only [FPGridBackward.reads,FPGridBackward.writes,FPGridBackward.address,FPGridWeight.address] at read
    simp at read
  · intro pid j
    rw [xf]
    · exact wo pid j
    · intro i
      rintro ⟨lane,eq⟩
      exact (by decide : ("dw":RegionName)≠"dx") (congrArg Prod.fst eq)
  · intro addr noX noW
    exact (xf addr noX).trans (wf addr noW)

theorem carry_congr (M : Algebra α) (gemma : Bool) (x g r x' g' r' : Nat → α) (n : Nat)
    (hx : ∀ i,i<n → x i=x' i) (hg : ∀ i,i<n → g i=g' i) (hr : ∀ i,i<n → r i=r' i) :
    FPWeightReference.carry M gemma x g r n=FPWeightReference.carry M gemma x' g' r' n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [FPWeightReference.carry]
    rw [ih (fun i hi => hx i (by omega)) (fun i hi => hg i (by omega)) (fun i hi => hr i (by omega)),
      hx n (by omega),hg n (by omega),hr n (by omega)]

def backwardRun (W X : Algebra α) (h : Call) (zero : Cell α) (initial : FPGridForward.Heap α) : Option (FPGridForward.Heap α) :=
  if h.rows=0 then some (zeroWeight h zero initial) else sequence W X h initial

theorem backward {β : Type} (W X : Algebra α) (h : Call) (zero : Cell α) (storage : Placement h .backward)
    (tensors : RegionName → VeriTile.Torch.TensorMetadata) (binding : Binding h .backward storage tensors)
    (heap : Nat → β) (decode : (Fin 4 → β) → Cell α) (other : Cell α)
    (logical : RegionName → Nat → α) (reads : ReadLaw .backward tensors heap decode (Cell.read .real) logical) :
    let initial:=lift storage heap decode other
    ∃ final,backwardRun W X h zero initial=some final ∧
      (∀ i : Fin h.rows,∀ j : Fin h.cols,final ("dx",i.val*h.cols+j.val)=inputReference X h logical i j) ∧
      (∀ j : Fin h.cols,final ("dw",j.val)=if h.rows=0 then zero else weightReference W h logical j) ∧
      Frame h .backward initial final := by
  dsimp only
  let initial:=lift storage heap decode other
  by_cases empty : h.rows=0
  · refine ⟨zeroWeight h zero initial,by simp [backwardRun,empty,initial],?_,?_,zeroWeight_frame h zero initial⟩
    · intro i;have hi:=i.isLt;omega
    · intro j;simp [zeroWeight,empty,j.isLt]
  have lr:=logical_read h .backward storage tensors binding heap decode (Cell.read .real) logical reads other
  obtain ⟨final,run,dx,dw,frame⟩:=sequence_complete W X h initial
  refine ⟨final,by simpa only [backwardRun,if_neg empty] using run,?_,?_,frame⟩
  · intro i j
    have rr (role : RegionName) (hr : role="x" ∨ role="dx") :
        FPReference.padded X h (FPMemory.raw h (entry initial i.val h.rows) role h.cols)=
          FPReference.padded X h (row h logical role i) := by
      apply padded_agrees
      intro k hk
      apply lr role (by rcases hr with rfl|rfl <;> simp [inputs])
      rcases hr with rfl|rfl <;> simpa [cells] using Native.offset_bound h i.val k.val i.isLt hk
    have wr : FPReference.padded X h (FPMemory.raw h (entry initial i.val h.rows) "w" 0)=
        FPReference.padded X h (weight h logical) := by
      apply padded_agrees
      intro k hk
      simpa [FPMemory.raw,entry,weight] using lr "w" (by simp [inputs]) k.val (by simpa [cells] using hk)
    have sr:=lr "r" (by simp [inputs]) i.val (by simpa [cells] using i.isLt)
    change (initial ("r",i.val)).read .real=logical "r" i.val at sr
    have value:=dx i j
    simp only [FPGridBackward.expected,inputReference,FPBackwardReference.output,FPBackwardReference.rowSum,
      FPBackwardReference.weighted,FPBackwardReference.normed,FPReference.input,rr "x" (Or.inl rfl),
      rr "dx" (Or.inr rfl),wr,sr] at value ⊢
    exact value
  · intro j
    let pid:=Geometry.owner h j
    let lane : FPGridWeight.Output h pid:=⟨Geometry.lane j,by simpa [pid,Geometry.covered] using j.isLt⟩
    have covered : Geometry.column pid.val lane.val=j.val := Geometry.covered h j
    have value:=dw pid lane
    have addr : FPGridWeight.address h pid lane=(("dw":RegionName),j.val) := Prod.ext rfl covered
    rw [addr] at value
    rw [value,if_neg empty]
    unfold FPGridWeight.expected weightReference FPWeightReference.output
    congr 1
    apply congrArg (FPReference.cast W)
    apply carry_congr
    · intro i hi
      simp only [FPWeightStep.raw,entry]
      have cov : pid.val*256+lane.val.val=j.val := covered
      rw [cov,if_pos j.isLt]
      exact lr "x" (by simp [inputs]) _ (by simpa [cells] using Native.offset_bound h i j.val hi j.isLt)
    · intro i hi
      simp only [FPWeightStep.raw,entry]
      have cov : pid.val*256+lane.val.val=j.val := covered
      rw [cov,if_pos j.isLt]
      exact lr "dx" (by simp [inputs]) _ (by simpa [cells] using Native.offset_bound h i j.val hi j.isLt)
    · intro i hi
      exact lr "r" (by simp [inputs]) i (by simpa [cells] using hi)

#axiomsClean backward
#specNonCircular forwardReference avoiding [FP.Structural.exec,Kernels.forward,Kernels.backward,Kernels.weightBackward]
#specNonCircular inputReference avoiding [FP.Structural.exec,Kernels.forward,Kernels.backward,Kernels.weightBackward]
#specNonCircular weightReference avoiding [FP.Structural.exec,Kernels.forward,Kernels.backward,Kernels.weightBackward]
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPTensor
