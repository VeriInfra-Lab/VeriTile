import bench.tritonbench_g.matmul_triton2.Proofs.SourceFullExecution
import bench.tritonbench_g.matmul_triton2.Proofs.Logical
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton2.SourceRuntime
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton2.Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def observedState (P : Core.Primitives α) (s : State (Word α)) : State α where
  mem := fun r o => mapCell (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P) (s.mem r o)
  regs := fun _ _ _ => none
  pids := s.pids
  numPids := s.numPids
  undef := fun d r o => mapValue (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P) d (s.undef d r o)

theorem observed_cell_read (P : Core.Primitives α) (cell : Cell (Word α)) :
    (mapCell (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P) cell).read .real =
      VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P (cell.read .real) := by
  cases cell with
  | mk d value => cases d <;> rfl

theorem observed_aValues (P : Core.Primitives α) (p : Core.Parameters) (s : State (Word α))
    (block : Nat) (i : Core.AIndex p) :
    VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P (Core.aValues (symbolicPrimitives α) p s block i) =
      Core.aValues P p (observedState P s) block i := by
  simp only [Core.aValues,Core.row,Core.rowBlock,Core.firstRow,Core.groupSize,Core.group,Core.width,Core.blocksM,Core.blocksN,observedState]
  split
  · exact (observed_cell_read P _).symm
  · rfl

theorem observed_bValues (P : Core.Primitives α) (p : Core.Parameters) (s : State (Word α))
    (block : Nat) (i : Core.BIndex p) :
    VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P (Core.bValues (symbolicPrimitives α) p s block i) =
      Core.bValues P p (observedState P s) block i := by
  simp only [Core.bValues,Core.column,Core.columnBlock,Core.firstRow,Core.groupSize,Core.group,Core.width,Core.blocksM,Core.blocksN,observedState]
  split
  · exact (observed_cell_read P _).symm
  · rfl

theorem observed_accumulation (P : Core.Primitives α) (p : Core.Parameters) (s : State (Word α))
    (block : Nat) (i : Core.CIndex p) :
    VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P (Core.accumulation (symbolicPrimitives α) p s block i) =
      Core.accumulation P p (observedState P s) block i := by
  induction block with
  | zero => rfl
  | succ block ih =>
    simp only [Core.accumulation,Core.matrixDot,symbolicPrimitives,VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe]
    simp only [symbolicPrimitives] at ih
    rw [ih]
    congr 1
    · funext k
      exact observed_aValues P p s block (i.1,k,PUnit.unit)
    · funext k
      exact observed_bValues P p s block (k,i.2.1,PUnit.unit)

theorem observed_logicalA (P : Core.Primitives α) (p : Core.Parameters) (s : State (Word α))
    (row block : Nat) (k : Fin p.blockInner) :
    VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P (Core.logicalA (symbolicPrimitives α) p s row block k) =
      Core.logicalA P p (observedState P s) row block k := by
  simp only [Core.logicalA,observedState]
  split
  · exact (observed_cell_read P _).symm
  · rfl

theorem observed_logicalB (P : Core.Primitives α) (p : Core.Parameters) (s : State (Word α))
    (col block : Nat) (k : Fin p.blockInner) :
    VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P (Core.logicalB (symbolicPrimitives α) p s col block k) =
      Core.logicalB P p (observedState P s) col block k := by
  simp only [Core.logicalB,observedState]
  split
  · exact (observed_cell_read P _).symm
  · rfl

theorem observed_logicalAccum (P : Core.Primitives α) (p : Core.Parameters) (s : State (Word α))
    (row col block : Nat) :
    VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P (Core.logicalAccum (symbolicPrimitives α) p s row col block) =
      Core.logicalAccum P p (observedState P s) row col block := by
  induction block with
  | zero => rfl
  | succ block ih =>
    simp only [Core.logicalAccum,symbolicPrimitives,VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe]
    simp only [symbolicPrimitives] at ih
    rw [ih]
    congr 1
    · funext k; exact observed_logicalA P p s row block k
    · funext k; exact observed_logicalB P p s col block k

theorem observed_logical (P : Core.Primitives α) (p : Core.Parameters) (s : State (Word α))
    (row col : Nat) :
    VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P (Core.logical (symbolicPrimitives α) p s row col) =
      Core.logical P p (observedState P s) row col := observed_logicalAccum P p s row col _

#axiomsClean observed_logicalA
#axiomsClean observed_logicalB
#axiomsClean observed_logicalAccum
#axiomsClean observed_logical

#axiomsClean observed_cell_read
#axiomsClean observed_aValues
#axiomsClean observed_bValues
#axiomsClean observed_accumulation
end
end VeriTile.Bench.Repairs.MatmulTriton2.SourceRuntime
