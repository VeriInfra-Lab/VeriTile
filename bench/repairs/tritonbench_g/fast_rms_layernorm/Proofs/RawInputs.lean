import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealTensor
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RuntimeForward

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealTensor
open VeriTile.Triton KernelTensor KernelStorage
open RealGridForward (read Heap)
noncomputable section

theorem forward_raw (h : Call) (initial : Heap)
    (logical : RegionName → Nat → ℝ)
    (reads : ∀ role, role∈inputs .forward → ∀ off, off<cells h role →
      read (initial (role,off))=logical role off) :
    ∃ final,RealGridForward.run h initial=some final ∧
      (∀ row : Fin h.rows,∀ j : Fin h.cols,read (final ("y",row.val*h.cols+j.val))=
        RealReference.output h.gemma h.eps (rowValues h logical "x" row) (weightValues h logical) j) ∧
      (∀ row : Fin h.rows,read (final ("r",row.val))=RealReference.inv h.eps (rowValues h logical "x" row)) ∧
      Frame h .forward initial final := by
  obtain ⟨final,run,values,frame⟩:=RealGridForward.complete h initial
  have xr (row : Fin h.rows) : RealRows.raw h (RealGridForward.entry initial row.val h.rows) "x" h.cols=
      rowValues h logical "x" row := by
    funext j
    apply reads "x" (by simp [inputs])
    simpa [cells] using Native.offset_bound h row.val j.val row.isLt j.isLt
  have wr (row : Fin h.rows) : RealRows.raw h (RealGridForward.entry initial row.val h.rows) "w" 0=
      weightValues h logical := by
    funext j
    simpa [RealRows.raw,RealGridForward.entry,weightValues,BlockState.readMem,RealGridForward.read] using
      reads "w" (by simp [inputs])
        j.val (by simpa [cells] using j.isLt)
  refine ⟨final,run,?_,?_,forward_frame h initial final frame⟩
  · intro row j
    simpa [RealGridForward.address,RealGridForward.expected,xr,wr] using values row (.inl j)
  · intro row
    simpa [RealGridForward.address,RealGridForward.expected,xr] using values row (.inr ())
theorem backward_raw (h : Call) (initial : Heap)
    (logical : RegionName → Nat → ℝ)
    (reads : ∀ role, role∈inputs .backward → ∀ off, off<cells h role →
      read (initial (role,off))=logical role off)
    (saved : ∀ row : Fin h.rows,logical "r" row.val=RealReference.inv h.eps (rowValues h logical "x" row)) :
    ∃ final,backwardRun h initial=some final ∧
      (∀ row : Fin h.rows,∀ j : Fin h.cols,read (final ("dx",row.val*h.cols+j.val))=
        RealReference.inputGradient h.gemma h.eps (rowValues h logical "x" row)
          (weightValues h logical) (rowValues h logical "dx" row) j) ∧
      (∀ j : Fin h.cols,read (final ("dw",j.val))=
        RealReference.weightGradient h.eps (rowValues h logical "x") (rowValues h logical "dx") j) ∧
      Frame h .backward initial final := by
  by_cases empty : h.rows=0
  · refine ⟨zeroWeight h (.real 0) initial,by simp [backwardRun,empty],?_,?_,zeroWeight_frame h (.real 0) initial⟩
    · intro i;have hi:=i.isLt;omega
    · intro j
      have cell : zeroWeight h (.real 0) initial ("dw",j.val)=MemCell.real 0 := by simp [zeroWeight,j.isLt]
      rw [cell]
      change 0=RealReference.weightGradient h.eps (rowValues h logical "x") (rowValues h logical "dx") j
      symm
      apply Finset.sum_eq_zero
      intro i _
      have hi:=i.isLt
      omega
  have lr (role : RegionName) (hr : role∈inputs .backward) (off : Nat) (hi : off<cells h role) :
      read (initial (role,off))=logical role off :=
    reads role hr off hi
  have data (role : RegionName) (hr : role="x" ∨ role="dx") :
      RealSequence.data h initial role=rowValues h logical role := by
    funext row j
    apply lr role (by rcases hr with rfl|rfl <;> simp [inputs])
    rcases hr with rfl|rfl <;> simpa [cells] using Native.offset_bound h row.val j.val row.isLt j.isLt
  have inv (row : Fin h.rows) : read (initial ("r",row.val))=RealReference.inv h.eps (RealSequence.data h initial "x" row) := by
    rw [lr "r" (by simp [inputs]) row.val (by simpa [cells] using row.isLt),saved,data "x" (Or.inl rfl)]
  obtain ⟨final,run,dx,dw,frame⟩:=RealSequence.complete h initial
  refine ⟨final,by simpa only [backwardRun,if_neg empty] using run,?_,?_,backward_frame h initial final frame⟩
  · intro row j
    have got:=dx row j
    unfold RealGridBackward.expected at got
    rw [lr "r" (by simp [inputs]) row.val (by simpa [cells] using row.isLt),saved] at got
    apply got.trans
    apply RealReads.gradient_agrees
    · intro k hk
      apply lr "x" (by simp [inputs])
      simpa [cells] using Native.offset_bound h row.val k.val row.isLt hk
    · intro k hk
      simpa [RealGridBackward.raw,weightValues] using lr "w" (by simp [inputs]) k.val (by simpa [cells] using hk)
    · intro k hk
      apply lr "dx" (by simp [inputs])
      simpa [cells] using Native.offset_bound h row.val k.val row.isLt hk
  · intro j
    let pid:=Geometry.owner h j
    let lane : RealGridWeight.Output h pid:=⟨Geometry.lane j,by simpa [pid,Geometry.covered] using j.isLt⟩
    have value:=dw pid lane
    have formula:=RealSequence.weight_formula h initial inv pid lane
    rw [data "x" (Or.inl rfl),data "dx" (Or.inr rfl)] at formula
    have indexEq : (⟨Geometry.column pid.val lane.val,lane.property⟩:Fin h.cols)=j := Fin.ext (Geometry.covered h j)
    rw [indexEq] at formula
    have addrEq : RealGridWeight.address h pid lane=(("dw":RegionName),j.val) := Prod.ext rfl (Geometry.covered h j)
    rw [addrEq] at value
    exact value.trans formula

#axiomsClean forward_raw
#axiomsClean backward_raw
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealTensor

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RuntimeForward.Tensor
open VeriTile.Triton FP.Structural KernelTensor KernelStorage FPTensor
open FPGridForward (Heap entry)
noncomputable section
variable {α : Type} [Inhabited α]

theorem forward_raw (M : Algebra α) (h : Call) (epsilon : α) (initial : FPGridForward.Heap α)
    (logical : RegionName → Nat → α)
    (reads : ∀ role, role∈inputs .forward → ∀ off, off<cells h role →
      (Cell.read .real) (initial (role,off))=logical role off) :
    ∃ final,Grid.run M h epsilon initial=some final ∧
      (∀ i : Fin h.rows,∀ j : Fin h.cols,final ("y",i.val*h.cols+j.val)=forwardReference M h epsilon logical i j) ∧
      (∀ i : Fin h.rows,final ("r",i.val)=savedReference M h epsilon logical i) ∧
      Frame h .forward initial final := by
  have lr:=reads
  obtain ⟨final,run,values,frame⟩:=Grid.complete M h epsilon initial
  have xr (i : Fin h.rows) : FPReference.padded M h (FPMemory.raw h (entry initial i.val h.rows) "x" h.cols)=
      FPReference.padded M h (row h logical "x" i) := by
    apply padded_agrees
    intro j hj
    apply lr "x" (by simp [inputs])
    simpa [cells] using Native.offset_bound h i.val j.val i.isLt hj
  have wr (i : Fin h.rows) : FPReference.padded M h (FPMemory.raw h (entry initial i.val h.rows) "w" 0)=
      FPReference.padded M h (weight h logical) := by
    apply padded_agrees
    intro j hj
    simpa [FPMemory.raw,entry,weight] using lr "w" (by simp [inputs]) j.val (by simpa [cells] using hj)
  refine ⟨final,run,?_,?_,forward_frame h initial final frame⟩
  · intro i j
    have value:=values i (.inl j)
    simp only [Grid.expected,forwardReference,Reference.output,Reference.normalized,
      Reference.inv,FPReference.input,xr,wr] at value ⊢
    exact value
  · intro i
    have value:=values i (.inr ())
    simp only [Grid.expected,savedReference,Reference.savedInv,Reference.inv,FPReference.input,xr] at value ⊢
    exact value


#axiomsClean forward_raw
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RuntimeForward.Tensor

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPTensor
open VeriTile.Triton FP.Structural KernelTensor KernelStorage FPTensor
open FPGridForward (Heap entry)
noncomputable section
variable {α : Type} [Inhabited α]

theorem backward_raw (W X : Algebra α) (h : Call) (zero : Cell α) (initial : FPGridForward.Heap α)
    (logical : RegionName → Nat → α)
    (reads : ∀ role, role∈inputs .backward → ∀ off, off<cells h role →
      (Cell.read .real) (initial (role,off))=logical role off) :
    ∃ final,backwardRun W X h zero initial=some final ∧
      (∀ i : Fin h.rows,∀ j : Fin h.cols,final ("dx",i.val*h.cols+j.val)=inputReference X h logical i j) ∧
      (∀ j : Fin h.cols,final ("dw",j.val)=if h.rows=0 then zero else weightReference W h logical j) ∧
      Frame h .backward initial final := by
  by_cases empty : h.rows=0
  · refine ⟨zeroWeight h zero initial,by simp [backwardRun,empty],?_,?_,zeroWeight_frame h zero initial⟩
    · intro i;have hi:=i.isLt;omega
    · intro j;simp [zeroWeight,empty,j.isLt]
  have lr:=reads
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

#axiomsClean backward_raw
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPTensor

