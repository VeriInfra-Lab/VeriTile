import VeriTile.Triton.Memory.AlgorithmFillProgramIO
import VeriTile.Triton.Float.FillProgramIO
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.FillProgramIO
open VeriTile Triton
noncomputable section
set_option linter.unusedSimpArgs false

def copy : ComputeKernel := .mk ["x"] ["o"] [
  .store .real [] (.region "o" (.constNat 0))
    (.alg (.load .real (.region "x" (.constNat 3)) .none .zero)) .none]

def fill (v : ℝ) : AlgorithmFill := ⟨"x",3,2,.real v⟩

example (s : BlockState) (v : ℝ) :
    (AlgorithmFill.mk "x" 19 0 (.real v)).apply s=s := by simp

example (s : BlockState) (v : ℝ) :
    ((fill v).apply s).mem "x" 3 = .real v := by
  simpa only [fill,Nat.add_zero] using (fill v).apply_mem_inside s 0 (by change 0 < 2; decide)

example (s : BlockState) (v : ℝ) (r : RegionName) (i : Nat)
    (outside : r ≠ "x" ∨ i < 3 ∨ 5 ≤ i) : ((fill v).apply s).mem r i=s.mem r i := by
  apply AlgorithmFill.apply_mem_outside
  simp only [AlgorithmFill.Contains,fill]
  rintro ⟨same,lo,hi⟩
  rcases outside with hr|h|h
  · exact hr same
  · omega
  · omega

example (v : ℝ) : (fill v).Safe (fun _=>5) := by
  intro i hi
  change i<2 at hi
  change 3+i<5
  omega

theorem algorithm_reads_fill (s : BlockState) (v : ℝ) :
    ∃ t,exec copy.toAlgKernel ((fill v).apply s)=some t ∧ t.readMem "o" 0=v := by
  simp [copy,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,exec,stepStmts,stepStmt,evalOp,
    bind,Except.bind,Option.bind,fill,AlgorithmFill.apply,AlgorithmFill.Contains,BlockState.readMem,BlockState.writeMem,BlockState.writeCell,MemCell.readAs,MemCell.real,MemCell.of]

namespace Native
open FP.Structural

def fill {α : Type} (v : α) : FiniteFill α := ⟨"x",3,2,.mk .real v⟩
def job {α : Type} (M : Algebra α) : Invocation α :=
  {kernel:=copy,backend:=M,pids:=fun _=>0}

example {α : Type} (s : State α) (v : α) :
    (FiniteFill.mk "x" 19 0 (.mk .real v)).apply s=s := by simp

example {α : Type} (s : State α) (v : α) : ((fill v).apply s).mem "x" 3=.mk .real v := by
  simpa only [fill,Nat.add_zero] using (fill v).apply_mem_inside s 0 (by change 0 < 2; decide)

example {α : Type} (s : State α) (v : α) (r : RegionName) (i : Nat)
    (outside : r ≠ "x" ∨ i < 3 ∨ 5 ≤ i) : ((fill v).apply s).mem r i=s.mem r i := by
  apply FiniteFill.apply_mem_outside
  simp only [FiniteFill.Contains,fill]
  rintro ⟨same,lo,hi⟩
  rcases outside with hr|h|h
  · exact hr same
  · omega
  · omega

theorem source_reads_fill {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (v : α) :
    ∃ t,runFillSchedule [.fill (fill v),.source (job M)] s=some t ∧
      t.mem "o" 0=.mk .real v := by
  simp [runFillSchedule,FillStage.run,job,Invocation.run,Invocation.initial,bindRegisters,
    FP.Structural.exec,copy,ComputeKernel.surfaceBody,run,step,evalExpr,evalOp_unfold,store,Option.bind,
    fill,FiniteFill.apply,FiniteFill.Contains]

end Native
end
end VeriTile.Tests.FillProgramIO
