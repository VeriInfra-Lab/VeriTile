import bench.tritonbench_g.int_scaled_matmul.Proofs.BlockWordLogical
import bench.tritonbench_g.int_scaled_matmul.Proofs.SourcePreservation
import VeriTile.Meta.StatementAudit

/-! A partial K tile of the actual repaired signed8/int32 source must use its
explicit zero padding, even with nonzero undefined-value oracles. -/
namespace VeriTile.Tests.IntScaledMatmulPadding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul Revision001 BlockWord
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

private def p : Parameters := ⟨"a","b","c",1,1,1,1,1,1,1,1,1,1,1,2,1⟩
private def initial {α : Type} [Inhabited α] : State α where
  mem := fun r _ => .mk .int (if r="a" then 2 else if r="b" then 3 else 0)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d r _ => match d with
    | .int => if r="a" then 5 else if r="b" then 7 else 0
    | other => defaultValue other

private def lane : CIndex p := (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit)

/-- The complete repaired source writes 6, rather than 6+5*7, and preserves
all memory outside its single output cell. The rest of the FP algebra is opaque. -/
theorem repaired_source_uses_zero_padding {α : Type} [Inhabited α] (rest : Algebra α) :
    ∃ final, FP.Structural.exec (backend rest) (Kernels.unscaled p) initial = some final ∧
      final.mem "c" 0 = .mk .int (6 : Int) ∧
      (∀ (r : RegionName) (o : Nat), r≠"c" ∨ o≠0 → final.mem r o=initial.mem r o) := by
  letI : Subsingleton (CIndex p) := by
    dsimp [CIndex,p,TileIndex]
    infer_instance
  have active : BlockWord.active p (initial : State α) lane := by
    norm_num [BlockWord.active,rowBlock,columnBlock,rowsInGroup,firstRow,blocksM,blocksN,p,initial,lane]
  obtain ⟨final,ran,values,frame⟩ := correctness rest p (by decide) (initial : State α)
    (fun _ _ _ _ _ => Subsingleton.elim _ _)
  have formula : outputValue p (initial : State α) lane=6 := by
    rw [output_logical p (by decide) initial lane active]
    norm_num [logicalDot,logicalProduct,LogicalSum.sum,wrap,p,initial,lane,
      rowBlock,columnBlock,rowsInGroup,firstRow,blocksM,blocksN,Cell.read,
      show ("b" : String)≠"a" from by decide]
    decide
  have offset : outputOffset p (initial : State α) lane=0 := by
    norm_num [outputOffset,rowBlock,columnBlock,rowsInGroup,firstRow,blocksM,blocksN,p,initial,lane]
  refine ⟨final,?_,?_,?_⟩
  · rw [SourcePreservation.unscaled]
    exact ran
  · have hv := values lane active
    simpa only [show p.output="c" from rfl,offset,formula] using hv
  · intro r o outside
    apply frame r o
    rcases outside with different|different
    · exact Or.inl different
    · right
      intro i hi
      have same : i=lane := Subsingleton.elim _ _
      subst i
      simpa only [offset] using different

#axiomsClean repaired_source_uses_zero_padding
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.IntScaledMatmulPadding
