import bench.tritonbench_g.fast_rms_layernorm.Proofs.KernelFrame
import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealEndToEnd

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealTensor
open VeriTile.Triton KernelTensor KernelStorage
open RealGridForward (read Heap)
noncomputable section

abbrev forwardFrame (h : Call) (initial final : Heap) :=
  ∀ addr,(∀ row : Fin h.rows,¬RealGridForward.writes h row addr) → final addr=initial addr
abbrev backwardFrame (h : Call) (initial final : Heap) :=
  ∀ addr,(∀ row : Fin h.rows,¬RealGridBackward.writes h row addr) →
    (∀ pid : Fin (Geometry.featureCount h),¬RealGridWeight.writes h pid addr) → final addr=initial addr

theorem forward {β : Type} (h : Call) (storage : Placement h .forward)
    (tensors : RegionName → VeriTile.Torch.TensorMetadata) (binding : Binding h .forward storage tensors)
    (heap : Nat → β) (decode : (Fin 4 → β) → MemCell) (other : MemCell)
    (logical : RegionName → Nat → ℝ) (reads : ReadLaw .forward tensors heap decode read logical) :
    let initial:=lift storage heap decode other
    ∃ final,RealGridForward.run h initial=some final ∧
      (∀ row : Fin h.rows,∀ j : Fin h.cols,read (final ("y",row.val*h.cols+j.val))=
        RealReference.output h.gemma h.eps (rowValues h logical "x" row) (weightValues h logical) j) ∧
      (∀ row : Fin h.rows,read (final ("r",row.val))=RealReference.inv h.eps (rowValues h logical "x" row)) ∧
      Frame h .forward initial final := by
  dsimp only
  let initial:=lift storage heap decode other
  obtain ⟨final,run,values,frame⟩:=RealGridForward.complete h initial
  have xr (row : Fin h.rows) : RealRows.raw h (RealGridForward.entry initial row.val h.rows) "x" h.cols=
      rowValues h logical "x" row := by
    funext j
    apply logical_read h .forward storage tensors binding heap decode read logical reads other "x" (by simp [inputs])
    simpa [cells] using Native.offset_bound h row.val j.val row.isLt j.isLt
  have wr (row : Fin h.rows) : RealRows.raw h (RealGridForward.entry initial row.val h.rows) "w" 0=
      weightValues h logical := by
    funext j
    simpa [RealRows.raw,RealGridForward.entry,weightValues,BlockState.readMem,RealGridForward.read] using
      logical_read h .forward storage tensors binding heap decode read logical reads other "w" (by simp [inputs])
        j.val (by simpa [cells] using j.isLt)
  refine ⟨final,run,?_,?_,forward_frame h initial final frame⟩
  · intro row j
    simpa [RealGridForward.address,RealGridForward.expected,xr,wr] using values row (.inl j)
  · intro row
    simpa [RealGridForward.address,RealGridForward.expected,xr] using values row (.inr ())

def backwardRun (h : Call) (initial : Heap) : Option Heap :=
  if h.rows=0 then some (zeroWeight h (.real 0) initial) else RealSequence.run h initial

/-- The saved-r relation is precisely what a preceding successful forward
establishes. It is not a successful-run assumption. Current saved values can
also be interpreted without claiming the original forward's derivative. -/
theorem backward {β : Type} (h : Call) (storage : Placement h .backward)
    (tensors : RegionName → VeriTile.Torch.TensorMetadata) (binding : Binding h .backward storage tensors)
    (heap : Nat → β) (decode : (Fin 4 → β) → MemCell) (other : MemCell)
    (logical : RegionName → Nat → ℝ) (reads : ReadLaw .backward tensors heap decode read logical)
    (saved : ∀ row : Fin h.rows,logical "r" row.val=RealReference.inv h.eps (rowValues h logical "x" row)) :
    let initial:=lift storage heap decode other
    ∃ final,backwardRun h initial=some final ∧
      (∀ row : Fin h.rows,∀ j : Fin h.cols,read (final ("dx",row.val*h.cols+j.val))=
        RealReference.inputGradient h.gemma h.eps (rowValues h logical "x" row)
          (weightValues h logical) (rowValues h logical "dx" row) j) ∧
      (∀ j : Fin h.cols,read (final ("dw",j.val))=
        RealReference.weightGradient h.eps (rowValues h logical "x") (rowValues h logical "dx") j) ∧
      Frame h .backward initial final := by
  dsimp only
  let initial:=lift storage heap decode other
  by_cases empty : h.rows=0
  · refine ⟨zeroWeight h (.real 0) initial,by simp [backwardRun,empty,initial],?_,?_,zeroWeight_frame h (.real 0) initial⟩
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
    logical_read h .backward storage tensors binding heap decode read logical reads other role hr off hi
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

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealTensor
