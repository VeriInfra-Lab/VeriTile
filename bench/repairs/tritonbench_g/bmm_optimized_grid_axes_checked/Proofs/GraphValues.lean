import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.SourceGraph
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.Complete

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourceGraph
open VeriTile.Triton FP.Structural BmmOptimized
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

def observeState (p : Scalar α) (s : State (Node α)) : State α where
  mem := fun r i=>mapCell (observe p) (s.mem r i)
  regs := fun d sh name=>(s.regs d sh name).map (fun values j=>mapValue (observe p) d (values j))
  pids := s.pids
  numPids := s.numPids
  undef := fun d r i=>mapValue (observe p) d (s.undef d r i)

theorem read_observe (p : Scalar α) (c : Cell (Node α)) :
    (mapCell (observe p) c).read .real=observe p (c.read .real) := by
  cases c with
  | mk d value=>cases d <;> rfl

theorem fmaFold_observe (p : Scalar α) (a b : Nat → Node α) (count : Nat) (carry : Node α) :
    observe p (Execution.fmaFold (scalar p) a b count carry)=
      Execution.fmaFold p (fun i=>observe p (a i)) (fun i=>observe p (b i)) count (observe p carry) := by
  induction count with
  | zero=>rfl
  | succ count ih=>simpa only [Execution.fmaFold,scalar,observe] using congrArg (p.fma (observe p (a count)) (observe p (b count))) ih

theorem dotTile_observe (p : Scalar α) {m k n : Nat}
    (a : Values (Node α) .real [m,k]) (b : Values (Node α) .real [k,n])
    (old : Values (Node α) .real [m,n]) (i : TileIndex [m,n]) :
    observe p (Execution.dotTile (scalar p) a b old i)=
      Execution.dotTile p (fun j=>observe p (a j)) (fun j=>observe p (b j))
        (fun j=>observe p (old j)) i := by
  rw [Execution.dotTile,fmaFold_observe]
  congr 1 <;> funext j <;> split <;> rfl

theorem aValue_observe (p : Scalar α) (cfg : Source.Parameters) (s : State (Node α))
    (k : Nat) (i : Execution.AIndex cfg) :
    observe p (Execution.aValue (scalar p) cfg s k i)=
      Execution.aValue p cfg (observeState p s) k i := by
  unfold Execution.aValue
  change observe p (if _ then _ else _)=if
    (cfg.divM=Bool.true ∨ Execution.row cfg s i.1<cfg.M) ∧
      (cfg.divK=Bool.true ∨ Execution.inner cfg k i.2.1<cfg.K)
    then (mapCell (observe p) (s.mem cfg.A (Execution.aOffset cfg s k i))).read .real else p.zero
  split <;> simp only [read_observe,scalar,observe]

theorem bValue_observe (p : Scalar α) (cfg : Source.Parameters) (s : State (Node α))
    (k : Nat) (i : Execution.BIndex cfg) :
    observe p (Execution.bValue (scalar p) cfg s k i)=
      Execution.bValue p cfg (observeState p s) k i := by
  unfold Execution.bValue
  change observe p (if _ then _ else _)=if
    (cfg.divK=Bool.true ∨ Execution.inner cfg k i.1<cfg.K) ∧
      (cfg.divN=Bool.true ∨ Execution.column cfg s i.2.1<cfg.N)
    then (mapCell (observe p) (s.mem cfg.B (Execution.bOffset cfg s k i))).read .real else p.zero
  split <;> simp only [read_observe,scalar,observe]

/-- The graph carrier keeps the exact old live-carry recurrence, for arbitrary
input nodes and all padded zero operations; no FMA law or padding identity. -/
theorem accumulation_observe (p : Scalar α) (cfg : Source.Parameters)
    (s : State (Node α)) (k : Nat) (i : Execution.CIndex cfg) :
    observe p (Execution.accumulation (scalar p) cfg s k i)=
      Execution.accumulation p cfg (observeState p s) k i := by
  induction k generalizing i with
  | zero=>rfl
  | succ k ih=>
    rw [Execution.accumulation,dotTile_observe]
    simp only [aValue_observe,bValue_observe]
    change Execution.dotTile p _ _ _ i=Execution.dotTile p _ _ _ i
    congr 1
    funext j
    exact ih j

#axiomsClean accumulation_observe
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourceGraph
