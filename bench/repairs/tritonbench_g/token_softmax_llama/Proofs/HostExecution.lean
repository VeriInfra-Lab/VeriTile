import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.HostCore
namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Host
open VeriTile.Triton FP.Structural Grid
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def dimensions (c : Call) (a : Nat) := if a=0 then batches c else if a=1 then heads c else 1

theorem launch_dimensions {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (positive : 0 < p.width) (m : Memory α) (pid : PID) (dims : Nat→Nat) :
    (FP.Structural.exec M (kernel .fp32 p) {state m pid with numPids:=dims}).map State.mem=
      launch M p pid m := by
  rw [(execution M .fp32 p {state m pid with numPids:=dims} positive).property,
    Option.map_some,execution_memory,launch_complete M p positive]
  congr 1
  simp only [reference,fold_writes_mem]
  rfl

def runActual {α : Type} [Inhabited α] (M : Algebra α) (c : Call) :
    List PID→Memory α→Option (Memory α)
  | [],m => some m
  | pid::rest,m =>
    ((FP.Structural.exec M (kernel .fp32 (params c))
      {state m pid with numPids:=dimensions c}).map State.mem).bind (runActual M c rest)

theorem runActual_eq {α : Type} [Inhabited α] (M : Algebra α)
    (c : Call) (positive : 0 < width c) (work : List PID) (m : Memory α) :
    runActual M c work m=Grid.run M (params c) work m := by
  induction work generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    simp only [runActual,launch_dimensions M (params c) positive,Grid.run]
    congr 1
    funext
    apply ih

structure Result (α : Type) where
  returned : Option Nat
  output : Matrix
  memory : Memory α

def invoke {α : Type} [Inhabited α] (M : Algebra α) (c : Call)
    (work : List PID) (m : Memory α) : Option (Result α) :=
  if c.maximum=0 then some ⟨none,c.output,m⟩ else do
    let final ← runActual M c work m
    pure ⟨none,c.output,final⟩

def outputEvents {α : Type} [Inhabited α] (c : Call) (m : Memory α) :=
  events (params c) m (fullGrid (batches c) (heads c))

theorem events_exact {α : Type} [Inhabited α] (c : Call) (m : Memory α) (e : Event (params c)) :
    e∈outputEvents c m ↔ (e.1.1 < batches c ∧ e.1.2 < heads c) ∧ e.2.1.val < seq c m e.1.1 := by
  rw [outputEvents,Grid.mem_events (params c) m _ e,Grid.mem_fullGrid]
  simp only [active,count_seq]

theorem events_schedule {α : Type} [Inhabited α] (c : Call) (m : Memory α)
    (work : List PID) (legal : (fullGrid (batches c) (heads c)).Perm work)
    (e : Event (params c)) : e∈outputEvents c m ↔ e∈events (params c) m work := by
  rw [outputEvents,Grid.mem_events (params c) m _ e,Grid.mem_events (params c) m work e]
  exact and_congr_left (fun _ => legal.mem_iff)

theorem events_zero {α : Type} [Inhabited α] (c : Call) (m : Memory α)
    (work : List PID) (zero : c.maximum=0) : events (params c) m work=[] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro e _
  have h := e.2.1.isLt
  change e.2.1.val < width c at h
  simp [width,VeriTile.Bench.Repairs.SquareMatrix.block,zero] at h

def Frame {α : Type} [Inhabited α] (c : Call) (before after : Memory α) :=
  ∀ r o, (r≠c.output.region ∨ ∀ e∈outputEvents c before,o≠offset (params c) before e.1 e.2) →
    after r o=before r o

/-- Success is derived by executing every CTA with fresh registers and actual
grid dimensions. The caller supplies the output object; the Python return is None. -/
theorem whole_typed {α : Type} [Inhabited α] (M : Algebra α)
    (c : Call) (work : List PID) (m : Memory α) (layout : Layout c) (data : Data c m)
    (legal : (fullGrid (batches c) (heads c)).Perm work) :
    ∃ result, invoke M c work m=some result ∧ result.returned=none ∧ result.output=c.output ∧
      result.memory=(events (params c) m work).foldl (Grid.write (params c) m (values M (params c) m)) m ∧
      (∀ e∈outputEvents c m,
        result.memory c.output.region (offset (params c) m e.1 e.2)=
          .mk .fp32 (values M (params c) m e)) ∧ Frame c m result.memory := by
  by_cases zero : c.maximum=0
  · refine ⟨⟨none,c.output,m⟩,by simp [invoke,zero],rfl,rfl,?_,?_,?_⟩
    · simp [events_zero c m work zero]
    · intro e he
      have h := e.2.1.isLt
      change e.2.1.val < width c at h
      simp [width,VeriTile.Bench.Repairs.SquareMatrix.block,zero] at h
    · intro r o _; rfl
  · have positive := width_positive c zero
    obtain ⟨final,run,trace,readback,frame⟩ := Grid.whole M (params c) positive work m
      layout.separated (unique c m layout data work legal)
    refine ⟨⟨none,c.output,final⟩,?_,rfl,rfl,trace,?_,?_⟩
    · simp [invoke,zero,runActual_eq M c positive,run]
    · intro e he
      exact readback e ((events_schedule c m work legal e).mp he)
    · intro r o outside
      apply frame r o
      rcases outside with hr|ho
      · exact Or.inl hr
      · exact Or.inr (fun e he => ho e ((events_schedule c m work legal e).mpr he))

#axiomsClean launch_dimensions
#axiomsClean events_exact
#axiomsClean whole_typed
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Host
