import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.FPTraceGrid

namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.FPTuning
open VeriTile Triton FP.Structural FPReference
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

def run (A : Algebra α) (P G Y : RegionName) (d : Dimensions) :
    List (Fin 6) → State α → Option (State α)
  | [],s => some s
  | c::cs,s => (FPGrid.backwardRun A P G Y d (backwardConfig d c) s).bind (run A P G Y d cs)

def graph (A : Algebra α) (P G Y : RegionName) (d : Dimensions) :
    List (Fin 6) → State α → State α
  | [],s => s
  | c::cs,s => graph A P G Y d cs (FPGrid.backwardGraph A P G Y d (backwardConfig d c) s)

theorem complete (A : Algebra α) (P G Y : RegionName) (d : Dimensions)
    (cs : List (Fin 6)) (s ref : State α) (h : s.mem=ref.mem) :
    ∃ t, run A P G Y d cs s=some t ∧ t.mem=(graph A P G Y d cs ref).mem := by
  induction cs generalizing s ref with
  | nil => exact ⟨s,rfl,h⟩
  | cons c cs ih =>
    obtain ⟨mid,he,hm⟩ := FPGrid.backward_whole A P G Y d (backwardConfig d c) s ref h
    obtain ⟨t,ht,hv⟩ := ih mid (FPGrid.backwardGraph A P G Y d (backwardConfig d c) ref) hm
    exact ⟨t,by simp [run,he,ht],hv⟩

theorem frame (A : Algebra α) (P G Y : RegionName) (d : Dimensions)
    (hN : 0<d.N) (hK : 0<d.K) (cs : List (Fin 6)) (s : State α) :
    FPTrace.Frame Y (d.M*d.N*d.K) s (graph A P G Y d cs s) := by
  induction cs generalizing s with
  | nil => intro r o _; rfl
  | cons c cs ih =>
    intro r o hn
    exact (ih _ r o hn).trans (FPTrace.backward_frame A P G Y d (backwardConfig d c) hN hK
      (backwardConfig_positive d c).1 (backwardConfig_positive d c).2 (backwardConfig_one d c) s r o hn)

theorem graph_append (A : Algebra α) (P G Y : RegionName) (d : Dimensions)
    (a b : List (Fin 6)) (s : State α) :
    graph A P G Y d (a++b) s=graph A P G Y d b (graph A P G Y d a s) := by
  induction a generalizing s with
  | nil => rfl
  | cons c cs ih => exact ih _

theorem run_append (A : Algebra α) (P G Y : RegionName) (d : Dimensions)
    (a b : List (Fin 6)) (s : State α) :
    run A P G Y d (a++b) s=(run A P G Y d a s).bind (run A P G Y d b) := by
  induction a generalizing s with
  | nil => rfl
  | cons c cs ih =>
    simp only [List.cons_append,run,ih,Option.bind_assoc]

def tuned (A : Algebra α) (P G Y : RegionName) (d : Dimensions)
    (trials : List (Fin 6)) (choice : Fin 6) (s : State α) : Option (State α) := do
  let warmed ← run A P G Y d trials s
  FPGrid.backwardRun A P G Y d (backwardConfig d choice) warmed

theorem tuned_complete (A : Algebra α) (P G Y : RegionName) (d : Dimensions)
    (trials : List (Fin 6)) (choice : Fin 6) (s ref : State α) (h : s.mem=ref.mem) :
    ∃ t, tuned A P G Y d trials choice s=some t ∧
      t.mem=(graph A P G Y d (trials++[choice]) ref).mem := by
  obtain ⟨t,ht,hm⟩ := complete A P G Y d (trials++[choice]) s ref h
  refine ⟨t,?_,hm⟩
  simpa [tuned,run_append,run] using ht

#axiomsClean complete
#axiomsClean frame
#axiomsClean tuned_complete
#specNonCircular graph avoiding [tuned,run,FPGrid.backwardRun,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxFlaggems.FPTuning
