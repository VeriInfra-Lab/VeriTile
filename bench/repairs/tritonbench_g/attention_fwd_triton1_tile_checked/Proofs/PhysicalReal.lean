import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.TensorEffects
import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.RealOutput
import Mathlib.Analysis.SpecialFunctions.Sqrt

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

def realScale (c : Shape) : ℝ := 1/Real.sqrt (c.D:ℝ)
theorem realScale_ne_zero (c : Shape) : realScale c≠0 := by
  have h : (0:ℝ)<c.D := Nat.cast_pos.mpr c.dim_pos
  exact div_ne_zero one_ne_zero (ne_of_gt (Real.sqrt_pos.2 h))

structure CellFrame {A} (c : Shape) (abi : ABI c) (storeFlag : Bool)
    (before after : Cells A) : Prop where
  outside : ∀ address, ¬Physical.WordAt abi.o address →
    (storeFlag=Bool.false ∨ ¬Physical.WordAt abi.h address) → after address=before address
  inputs : ∀ a∈[abi.q,abi.k,abi.v], ∀ address, a.Contains (address/4) → after address=before address
  disabledHistory : storeFlag=Bool.false → ∀ address,
    abi.h.Contains (address/4) → after address=before address

namespace PhysicalCorrect

theorem frame {A} [Inhabited A] (ops : Ops A) (literal : ℝ → A)
    (c : Shape) (abi : ABI c) (scale : ℝ) (storeFlag ifcond : Bool)
    (before : Cells A) (rest : State (Term A)) :
    CellFrame c abi storeFlag before (Physical.result ops literal c abi scale storeFlag ifcond before rest) := by
  refine ⟨Physical.frame ops literal c abi scale storeFlag ifcond before rest,
    TensorEffect.input_frame ops literal c abi scale storeFlag ifcond before rest,?_⟩
  intro hs
  subst storeFlag
  exact TensorEffect.history_disabled ops literal c abi scale ifcond before rest

/-- Complete shared quote over raw ABI allocations and typed cells, with the
independent mathematical output/history and complete caller-cell frame. -/
theorem kernel_real (c : Shape) (abi : ABI c) (storeFlag ifcond : Bool)
    (before : Cells ℝ) (rest : State (Term ℝ)) :
    Physical.run realOps id c abi (realScale c) storeFlag ifcond before rest=
      some (Physical.result realOps id c abi (realScale c) storeFlag ifcond before rest) ∧
    (∀ head row j,head<c.heads → row<c.T → j<c.D →
      Physical.result realOps id c abi (realScale c) storeFlag ifcond before rest
        (4*(abi.o.base+c.mainOffset head row j))=
        .mk .fp32 (MathTarget.output c (tensorInputs c abi before) (realScale c) head row j)) ∧
    (storeFlag=Bool.true → ∀ head chunk i j,head<c.heads → chunk<c.chunks → i<c.D → j<c.D →
      Physical.result realOps id c abi (realScale c) storeFlag ifcond before rest
        (4*(abi.h.base+c.historyOffset head chunk i j))=
        .mk .fp32 (MathTarget.history c (tensorInputs c abi before) head chunk i j)) ∧
    CellFrame c abi storeFlag before (Physical.result realOps id c abi (realScale c) storeFlag ifcond before rest) := by
  refine ⟨Physical.success _ _ _ _ _ _ _ _ _,?_,?_,frame _ _ _ _ _ _ _ _ _⟩
  · intro head row j hh hr hj
    rw [Physical.output realOps id rfl c abi (realScale c) storeFlag ifcond before rest head row ⟨j,hj⟩ hh hr]
    change Cell.mk .fp32 (Reference.output realOps c (tensorInputs c abi before) (realScale c)
      head (row/32) ifcond (row%32) j)=_
    rw [RealProof.output_target c _ (realScale c) head row j ifcond hr hj]
  · intro hs head chunk i j hh hc hi hj
    subst storeFlag
    rw [Physical.history realOps id rfl c abi (realScale c) ifcond before rest head chunk ⟨i,hi⟩ ⟨j,hj⟩ hh hc]
    change Cell.mk .fp32 (Reference.carry realOps c (tensorInputs c abi before) (realScale c)
      head ifcond chunk i j)=_
    rw [RealProof.carry_target c _ (realScale c) head ifcond chunk i j hi hj]

#axiomsClean kernel_real
#specNonCircular MathTarget.output avoiding [FP.Structural.exec,Physical.run,Physical.result,Reference.output]
#specNonCircular MathTarget.history avoiding [FP.Structural.exec,Physical.run,Physical.result,Reference.carry]
end PhysicalCorrect
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
