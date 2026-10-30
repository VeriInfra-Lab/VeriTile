import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealGridBackward
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealGridWeight

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealSequence
open VeriTile.Triton
open RealGridForward (Address Heap read)
noncomputable section
open scoped BigOperators

def run (h : Call) (heap : Heap) : Option Heap :=
  (RealGridWeight.run h heap).bind (RealGridBackward.run h)

theorem complete (h : Call) (heap : Heap) :
    ∃ final,run h heap=some final ∧
      (∀ row : Fin h.rows,∀ j,read (final (RealGridBackward.address h row j))=RealGridBackward.expected h heap row j) ∧
      (∀ pid : Fin (Geometry.featureCount h),∀ j,read (final (RealGridWeight.address h pid j))=RealGridWeight.expected h heap pid j) ∧
      (∀ addr,(∀ row : Fin h.rows,¬RealGridBackward.writes h row addr) →
        (∀ pid : Fin (Geometry.featureCount h),¬RealGridWeight.writes h pid addr) → final addr=heap addr) := by
  obtain ⟨weights,hw,wo,wf⟩ := RealGridWeight.complete h heap
  obtain ⟨final,hx,xo,xf⟩ := RealGridBackward.complete h weights
  refine ⟨final,by simp [run,hw,hx],?_,?_,?_⟩
  · intro row j
    rw [xo]
    apply RealGridBackward.expected_stable h row weights heap
    intro addr read
    apply wf
    intro pid
    rintro ⟨lane,eq⟩
    subst addr
    simp only [RealGridBackward.reads,RealGridBackward.writes,RealGridBackward.address,
      RealGridWeight.address] at read
    simp at read
  · intro pid j
    rw [xf]
    · exact wo pid j
    · intro row
      rintro ⟨lane,eq⟩
      exact (by decide : ("dw":RegionName)≠"dx") (congrArg Prod.fst eq)
  · intro addr noX noW
    exact (xf addr noX).trans (wf addr noW)

def data (h : Call) (heap : Heap) (role : RegionName) (row : Fin h.rows) (j : Fin h.cols) : ℝ :=
  read (heap (role,row.val*h.cols+j.val))

theorem weight_formula (h : Call) (heap : Heap)
    (saved : ∀ row : Fin h.rows,read (heap ("r",row.val))=RealReference.inv h.eps (data h heap "x" row))
    (pid : Fin (Geometry.featureCount h)) (j : RealGridWeight.Output h pid) :
    RealGridWeight.expected h heap pid j=RealReference.weightGradient h.eps (data h heap "x") (data h heap "dx")
      ⟨Geometry.column pid.val j.val,j.property⟩ := by
  change (∑ row∈Finset.range h.rows,
    (if Geometry.column pid.val j.val<h.cols then read (heap ("dx",row*h.cols+Geometry.column pid.val j.val)) else 0)*
      ((if Geometry.column pid.val j.val<h.cols then read (heap ("x",row*h.cols+Geometry.column pid.val j.val)) else 0)*
        read (heap ("r",row))))=_
  simp only [if_pos j.property]
  rw [←Fin.sum_univ_eq_sum_range]
  unfold RealReference.weightGradient
  apply Finset.sum_congr rfl
  intro row _
  rw [saved row]
  dsimp only [data]
  ring

#axiomsClean complete
#axiomsClean weight_formula
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealSequence
