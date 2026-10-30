import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.FPTuning
namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.FPTuning
open VeriTile Triton FP.Structural FPTrace
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

def trace (A : Algebra α) (P G Y : RegionName) (d : Dimensions) :
    List (Fin 6) → State α → List (Tile α)
  | [],_ => []
  | c::cs,s => backwardTrace A P G Y d (backwardConfig d c) s ++
      trace A P G Y d cs (FPGrid.backwardGraph A P G Y d (backwardConfig d c) s)

theorem graph_trace (A : Algebra α) (P G Y : RegionName) (d : Dimensions)
    (cs : List (Fin 6)) (s : State α) : graph A P G Y d cs s=execute (trace A P G Y d cs s) s := by
  induction cs generalizing s with
  | nil => rfl
  | cons c cs ih =>
    rw [graph,trace,execute_append,←backward_graph_trace,←ih]

theorem events_bounds (A : Algebra α) (P G Y : RegionName) (d : Dimensions)
    (hN : 0<d.N) (hK : 0<d.K) (cs : List (Fin 6)) (s : State α) (a : RegionName × Nat)
    (ha : a∈events (trace A P G Y d cs s)) : a.1=Y ∧ a.2<d.M*d.N*d.K := by
  induction cs generalizing s with
  | nil => simp [trace,events] at ha
  | cons c cs ih =>
    rcases (mem_events_append _ _ _).mp ha with h|h
    · exact (backward_events_exact A P G Y d (backwardConfig d c) hN hK
        (backwardConfig_positive d c).1 (backwardConfig_positive d c).2 (backwardConfig_one d c) s a).mp h
    · exact ih _ h

theorem events_exact (A : Algebra α) (P G Y : RegionName) (d : Dimensions)
    (hN : 0<d.N) (hK : 0<d.K) (cs : List (Fin 6)) (hne : cs≠[]) (s : State α) (a : RegionName × Nat) :
    a∈events (trace A P G Y d cs s) ↔ a.1=Y ∧ a.2<d.M*d.N*d.K := by
  refine ⟨events_bounds A P G Y d hN hK cs s a,?_⟩
  intro ha
  cases cs with
  | nil => exact (hne rfl).elim
  | cons c cs =>
    rw [trace,mem_events_append]
    exact Or.inl ((backward_events_exact A P G Y d (backwardConfig d c) hN hK
      (backwardConfig_positive d c).1 (backwardConfig_positive d c).2 (backwardConfig_one d c) s a).mpr ha)

#axiomsClean graph_trace
#axiomsClean events_exact
end
end VeriTile.Bench.Repairs.SoftmaxFlaggems.FPTuning
