import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.PhysicalReal
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.NumericalReference

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

namespace Native
/-- Native binding of the quote's two scalar literal sites. Zero has the
FP32 +0 word; the nonzero scale slot uses the actual host/JIT scale word.
No equality between this word and an ideal real scale is asserted. -/
def literal (actualScale : Word) (x : ℝ) : Word := if x=0 then 0 else actualScale
@[simp] theorem literal_zero (actualScale : Word) : literal actualScale 0=0 := by simp [literal]
@[simp] theorem literal_scale (c : Shape) (actualScale : Word) :
    literal actualScale (realScale c)=actualScale := by simp [literal,realScale_ne_zero]

def result (native : ISA) (c : Shape) (abi : ABI c) (actualScale : Word)
    (storeFlag ifcond : Bool) (before : Cells Word) (rest : State (Term Word)) : Cells Word :=
  Physical.result (wordOps native) (literal actualScale) c abi (realScale c) storeFlag ifcond before rest

end Native

namespace PhysicalFPEquiv

/-- Complete shared FP32 quote and raw output/history word observations.
Only the unchanged local mulRN/fmaRN laws and reached domains are numerical premises. -/
theorem kernel_native (native : ISA) (dom : NumericDomain) (c : Shape) (abi : ABI c)
    (actualScale : Word) (storeFlag ifcond : Bool) (before : Cells Word) (rest : State (Term Word))
    (numeric : NumericalReference.BitNumerical native dom c (tensorInputs c abi before) actualScale ifcond) :
    Physical.run (wordOps native) (Native.literal actualScale) c abi (realScale c)
      storeFlag ifcond before rest=some (Native.result native c abi actualScale storeFlag ifcond before rest) ∧
    (∀ head row j,head<c.heads → row<c.T → j<c.D →
      Native.result native c abi actualScale storeFlag ifcond before rest (4*(abi.o.base+c.mainOffset head row j))=
        .mk .fp32 (NumericalReference.SourceFP.output c (tensorInputs c abi before) actualScale head row j ifcond)) ∧
    (storeFlag=Bool.true → ∀ head chunk i j,head<c.heads → chunk<c.chunks → i<c.D → j<c.D →
      Native.result native c abi actualScale storeFlag ifcond before rest (4*(abi.h.base+c.historyOffset head chunk i j))=
        .mk .fp32 (NumericalReference.SourceFP.history c (tensorInputs c abi before) actualScale head chunk i j ifcond)) ∧
    CellFrame c abi storeFlag before (Native.result native c abi actualScale storeFlag ifcond before rest) := by
  refine ⟨Physical.success _ _ _ _ _ _ _ _ _,?_,?_,PhysicalCorrect.frame _ _ _ _ _ _ _ _ _⟩
  · intro head row j hh hr hj
    unfold Native.result
    rw [Physical.output (wordOps native) (Native.literal actualScale) (by simp [wordOps])
      c abi (realScale c) storeFlag ifcond before rest head row ⟨j,hj⟩ hh hr]
    change Cell.mk .fp32 (Reference.output (wordOps native) c (tensorInputs c abi before)
      (Native.literal actualScale (realScale c)) head (row/32) ifcond (row%32) j)=_
    rw [Native.literal_scale]
    rw [←AttentionFwdTriton1.Whole001.Graph.eval_output native c (tensorInputs c abi before) actualScale
      head (row/32) ifcond (row%32) j]
    rw [NumericalReference.SourceFP.transport native dom numeric.scalarLaws _
      (numeric.admitted.output head (row/32) (row%32) j hh (c.chunk_of_row row hr)
        (Nat.mod_lt row (by decide)) hj),AttentionFwdTriton1.Whole001.Graph.eval_output]
    rfl
  · intro hs head chunk i j hh hc hi hj
    subst storeFlag
    unfold Native.result
    rw [Physical.history (wordOps native) (Native.literal actualScale) (by simp [wordOps])
      c abi (realScale c) ifcond before rest head chunk ⟨i,hi⟩ ⟨j,hj⟩ hh hc]
    change Cell.mk .fp32 (Reference.carry (wordOps native) c (tensorInputs c abi before)
      (Native.literal actualScale (realScale c)) head ifcond chunk i j)=_
    rw [Native.literal_scale]
    rw [←AttentionFwdTriton1.Whole001.Graph.eval_carry native c (tensorInputs c abi before) actualScale head ifcond chunk i j]
    rw [NumericalReference.SourceFP.transport native dom numeric.scalarLaws _
      (numeric.admitted.carry head chunk i j hh (Nat.le_of_lt hc) hi hj),AttentionFwdTriton1.Whole001.Graph.eval_carry]
    rfl

#axiomsClean kernel_native
#specNonCircular NumericalReference.SourceFP.output avoiding [FP.Structural.exec,Physical.run,Physical.result,Native.result]
#specNonCircular NumericalReference.SourceFP.history avoiding [FP.Structural.exec,Physical.run,Physical.result,Native.result]
end PhysicalFPEquiv
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
