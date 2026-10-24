import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.IOComplete
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.GraphValues

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Interface
open VeriTile.Triton FP.Structural BmmOptimized SourceGraph
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

/-- Exact complete-grid correspondence to the original fused executor. The
graph initial memory may contain arbitrary input graphs; all original raw tags,
current-memory CTA execution and exterior cells are retained under observation. -/
theorem transport (s : Shape) (c : Config) (p : Scalar α) (rest : Algebra α)
    (ps : List Program) (order : ps.Perm (programs s c))
    (graph : State (Node α)) (initial : State α)
    (related : initial.mem=fun r i=>mapCell (observe p) (graph.mem r i))
    (after : State (Node α))
    (run : runSchedule (ps.map (invocation s c p rest)) graph=some after) :
    ∃ final,Bridge.grid s c p rest ps initial=some final ∧
      final.mem=fun r i=>mapCell (observe p) (after.mem r i) := by
  let m:=inputMemory graph
  have inputs : ReadGrid.Inputs (fun r i=>observe p (m r i)) initial := by
    constructor
    · intro i
      rw [related,read_observe]
      rfl
    · intro i
      rw [related,read_observe]
      rfl
  obtain ⟨final,oldRun,values,frame⟩:=ReadGrid.ordered_execution s c p rest ps order
    (fun r i=>observe p (m r i)) initial inputs
  obtain ⟨observed,graphRun,graphValues,graphFrame⟩:=ReadGrid.ordered_execution s c
    (scalar p) (base p rest) ps order m graph ⟨fun _=>rfl,fun _=>rfl⟩
  rw [schedule_equal] at run
  have eq : after=observed := Option.some.inj (run.symm.trans graphRun)
  subst observed
  refine ⟨final,oldRun,?_⟩
  funext r i
  by_cases region : r="O"
  · subst r
    by_cases bound : i<s.size
    · let j : Fin s.size:=⟨i,bound⟩
      have valid:=output_valid s j
      have old:=values (outputCell s j) valid.1
      have fresh:=graphValues (outputCell s j) valid.1
      rw [valid.2] at old fresh
      rw [old,fresh]
      change FP.Structural.Cell.mk .real (reference s p (fun r i=>observe p (m r i)) _ _)=
        .mk .real (observe p (reference s (scalar p) m _ _))
      rw [reference_observe]
    · have outside : s.size ≤ i := by omega
      rw [frame _ _ (Or.inr outside),graphFrame _ _ (Or.inr outside),related]
  · rw [frame _ _ (Or.inl region),graphFrame _ _ (Or.inl region),related]

def liftState (s : State α) : State (Node α) where
  mem := fun r i=>mapCell Node.value (s.mem r i)
  regs := fun d sh name=>(s.regs d sh name).map (fun values j=>mapValue Node.value d (values j))
  pids := s.pids
  numPids := s.numPids
  undef := fun d r i=>mapValue Node.value d (s.undef d r i)

theorem lift_observe (p : Scalar α) (initial : State α) :
    initial.mem=fun r i=>mapCell (observe p) ((liftState initial).mem r i) := by
  funext r i
  change initial.mem r i=mapCell (observe p) (mapCell Node.value (initial.mem r i))
  cases initial.mem r i with
  | mk d value=>cases d <;> rfl

#axiomsClean transport
#axiomsClean lift_observe
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Interface
