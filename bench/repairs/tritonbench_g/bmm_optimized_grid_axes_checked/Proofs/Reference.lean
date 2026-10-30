import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.Execution
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.SourceDecomposition

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Execution
open VeriTile.Triton FP.Structural Source
noncomputable section
abbrev AIndex (p : Parameters) := TileIndex [p.TM,p.TK]
abbrev BIndex (p : Parameters) := TileIndex [p.TK,p.TN]
abbrev CIndex (p : Parameters) := TileIndex [p.TM,p.TN]

def rowBlock (p : Parameters) (s : RegisterState α) :=
  VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.rowTile
    (s.pids 0) (s.pids 1) (s.numPids 0) (s.numPids 1) p.GROUP
def columnBlock (p : Parameters) (s : RegisterState α) :=
  VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid.columnTile
    (s.pids 0) (s.pids 1) (s.numPids 0) (s.numPids 1) p.GROUP
def row (p : Parameters) (s : RegisterState α) (i : Fin p.TM) := rowBlock p s*p.TM+i.val
def column (p : Parameters) (s : RegisterState α) (i : Fin p.TN) := columnBlock p s*p.TN+i.val
def inner (p : Parameters) (step : Nat) (i : Fin p.TK) := step*p.TK+i.val

def aOffset (p : Parameters) (s : RegisterState α) (step : Nat) (i : AIndex p) :=
  s.pids 2*p.M*p.K+row p s i.1*p.K+i.2.1.val+step*p.TK
def bOffset (p : Parameters) (s : RegisterState α) (step : Nat) (i : BIndex p) :=
  s.pids 2*p.K*p.N+i.1.val*p.N+column p s i.2.1+step*(p.TK*p.N)
def oOffset (p : Parameters) (s : RegisterState α) (i : CIndex p) :=
  s.pids 2*p.M*p.N+row p s i.1*p.N+column p s i.2.1

def aValue [Inhabited α] (o : BmmOptimized.Scalar α) (p : Parameters)
    (s : RegisterState α) (step : Nat) (i : AIndex p) : α :=
  if (p.divM=Bool.true ∨ row p s i.1<p.M) ∧
      (p.divK=Bool.true ∨ inner p step i.2.1<p.K) then
    (s.mem p.A (aOffset p s step i)).read .real else o.zero
def bValue [Inhabited α] (o : BmmOptimized.Scalar α) (p : Parameters)
    (s : RegisterState α) (step : Nat) (i : BIndex p) : α :=
  if (p.divK=Bool.true ∨ inner p step i.1<p.K) ∧
      (p.divN=Bool.true ∨ column p s i.2.1<p.N) then
    (s.mem p.B (bOffset p s step i)).read .real else o.zero

def accumulation [Inhabited α] (o : BmmOptimized.Scalar α) (p : Parameters)
    (s : RegisterState α) : Nat → CIndex p → α
  | 0 => fun _ => o.zero
  | step+1 => dotTile o (aValue o p s step) (bValue o p s step) (accumulation o p s step)
def active (p : Parameters) (s : RegisterState α) (i : CIndex p) : Prop :=
  (p.divM=Bool.true ∨ row p s i.1<p.M) ∧ (p.divN=Bool.true ∨ column p s i.2.1<p.N)
instance (p : Parameters) (s : RegisterState α) : DecidablePred (active p s) :=
  fun _=>inferInstanceAs (Decidable (_ ∧ _))

def iterations (p : Parameters) := (p.K+p.TK-1)/p.TK
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Execution
