import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.FP
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.NativeGrid

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
open VeriTile.Triton FP.Structural Whole
noncomputable section
set_option maxHeartbeats 4000000

def withConfig (r : Request) (config : Config) : Request := {r with config := config}

@[simp] theorem withConfig_self (r : Request) : withConfig r r.config=r := by
  cases r
  rfl

/-- Full memory frame, including the unused part of the output allocation. -/
def OutputFrame {α : Type} (r : Request) (before after : State α) : Prop :=
  ∀ region offset, ¬Footprint r region offset → after.mem region offset=before.mem region offset

theorem OutputFrame.inputs {α : Type} (r : Request) (before after : State α)
    (frame : OutputFrame r before after) (hA : r.C≠r.A) (hB : r.C≠r.B) :
    (∀ o, after.mem r.A o=before.mem r.A o) ∧
    (∀ o, after.mem r.B o=before.mem r.B o) := by
  constructor
  · intro o
    exact frame r.A o (fun h => hA h.1.symm)
  · intro o
    exact frame r.B o (fun h => hB h.1.symm)

/-- The per-grid frame does not depend on any numerical law. -/
theorem programs_frame {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (r : Request)
    (ids : List Nat) (initial : State α) :
    ∃ final, runPrograms M D r ids initial=some final ∧ OutputFrame r initial final := by
  induction ids generalizing initial with
  | nil => exact ⟨initial,rfl,fun _ _ _ => rfl⟩
  | cons pid rest ih =>
    obtain ⟨old,_,hm,_,hf⟩ := FPValue.correctness M D r.parameters .single
      (instanceState r pid initial) (Whole.active_injective r (instanceState r pid initial))
    obtain ⟨next,he,hn⟩ := Source.execution_memory M D r.parameters (instanceState r pid initial)
    have hmem : next.mem=old.mem := hn.trans hm.symm
    have hframe : OutputFrame r initial next := by
      intro region offset hout
      rw [hmem]
      apply hf
      by_cases hr : region=r.C
      · right
        intro i hi heq
        apply hout
        exact ⟨hr,FPValue.row r.parameters (instanceState r pid initial) i.1,hi.1,
          FPValue.col r.parameters (instanceState r pid initial) i.2.1,hi.2,
          heq.trans (Whole.source_offset r _ i)⟩
      · exact Or.inl hr
    obtain ⟨final,hfinal,hrest⟩ := ih next
    refine ⟨final,?_,fun region offset hout => (hrest region offset hout).trans (hframe region offset hout)⟩
    change (FP.Structural.exec (FPValue.backend M D) (Source.kernel r.parameters)
      (instanceState r pid initial)).bind (runPrograms M D r rest)=some final
    rw [he]
    exact hfinal

/-- Arbitrarily many benchmark grids, including repeats and a cache-hit empty
list. Each starts from the state left by its predecessor. -/
def runTrials {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) : List Config → State α → Option (State α)
  | [],s => some s
  | config::rest,s =>
    (runPrograms M D (withConfig r config) (List.range (programs (withConfig r config))) s).bind
      (runTrials M D r rest)

theorem trials_frame {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (r : Request)
    (configs : List Config) (initial : State α) :
    ∃ final, runTrials M D r configs initial=some final ∧ OutputFrame r initial final := by
  induction configs generalizing initial with
  | nil => exact ⟨initial,rfl,fun _ _ _ => rfl⟩
  | cons config rest ih =>
    obtain ⟨next,he,hf⟩ := programs_frame M D (withConfig r config)
      (List.range (programs (withConfig r config))) initial
    obtain ⟨final,hfinal,hrest⟩ := ih next
    exact ⟨final,by simp [runTrials,he,hfinal],
      fun region offset hout => (hrest region offset hout).trans (hf region offset hout)⟩

theorem runTrials_append {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (r : Request)
    (xs ys : List Config) (s : State α) :
    runTrials M D r (xs++ys) s=(runTrials M D r xs s).bind (runTrials M D r ys) := by
  induction xs generalizing s with
  | nil => rfl
  | cons x xs ih => simp only [List.cons_append,runTrials,ih,Option.bind_assoc]

def TensorAPI.reconfigure {α : Type} {r : Request} {s : State α}
    (t : TensorAPI r s) (config : Config) (legal : config∈configurations r.target) :
    TensorAPI (withConfig r config) s := {t with selected := legal}

def TensorAPI.afterTrials {α : Type} {r : Request} {s u : State α}
    (t : TensorAPI r s) (frame : OutputFrame r s u) : TensorAPI r u :=
  let same := frame.inputs r s u t.freshA t.freshB
  {t with
    tagA := by intro row hr k hk; rw [same.1]; exact t.tagA row hr k hk
    tagB := by intro k hk col hc; rw [same.2]; exact t.tagB k hk col hc
    aliasConsistency := by
      intro row hr k hk k' hk' col hc he
      rw [same.1,same.2]
      exact t.aliasConsistency row hr k hk k' hk' col hc he}

/-- Trials only change C; resolved logical inputs and the caller's complete
preparation frame therefore remain valid for the selected final grid. -/
def ResolveNegContract.afterTrials {α : Type} [Inhabited α]
    {r : Request} {before prepared after : State α} {t : TensorAPI r prepared}
    {A : Fin r.rows → Fin r.inner → α} {B : Fin r.inner → Fin r.columns → α}
    (resolve : ResolveNegContract r before prepared t A B)
    (frame : OutputFrame r prepared after) :
    ResolveNegContract r before after (t.afterTrials frame) A B :=
  let same := frame.inputs r prepared after t.freshA t.freshB
  {resolve with
    logicalA := by intro i k; unfold inputA; rw [same.1]; exact resolve.logicalA i k
    logicalB := by intro k j; unfold inputB; rw [same.2]; exact resolve.logicalB k j
    frame := by
      intro region offset hout hcopy
      exact (frame region offset hout).trans (resolve.frame region offset hout hcopy)}

namespace Native
def runConfigs {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (b : Bases) : List Config → State α → Option (State α)
  | [],s => some s
  | config::rest,s =>
    (runPrograms M D (withConfig r config) b (List.range (programs (withConfig r config))) s).bind
      (runConfigs M D r b rest)

/-- The complete finite native sequence executes with the same memory as the
typed sequence. No numerical or per-trial output assumption occurs here. -/
theorem configs_correspondence {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (r : Request)
    (prepared : State α) (t : TensorAPI r prepared) (configs : List Config)
    (legal : ∀ config∈configs, config∈configurations r.target)
    (abi : ∀ config∈configs, programs (withConfig r config)<2^31)
    (s u : State α) (same : s.mem=u.mem) :
    ∃ native typed, runConfigs M D r (bases t) configs s=some native ∧
      runTrials M D r configs u=some typed ∧ native.mem=typed.mem := by
  induction configs generalizing s u with
  | nil => exact ⟨s,u,rfl,rfl,same⟩
  | cons config rest ih =>
    let tc := t.reconfigure config (legal config (by simp))
    have hg := abi config (by simp)
    obtain ⟨n,t',hn,ht,hm⟩ := programs_correspondence M D (withConfig r config) prepared tc _
      (fun pid h => ⟨List.mem_range.mp h,lt_trans (List.mem_range.mp h) hg⟩) s u same
    change runPrograms M D (withConfig r config) (bases t)
      (List.range (programs (withConfig r config))) s=some n at hn
    obtain ⟨nf,tf,hnf,htf,hfinal⟩ := ih
      (fun c h => legal c (List.mem_cons_of_mem config h))
      (fun c h => abi c (List.mem_cons_of_mem config h)) n t' hm
    exact ⟨nf,tf,by simp only [runConfigs]; rw [hn]; exact hnf,
      by simp only [runTrials]; rw [ht]; exact htf,hfinal⟩
end Native

/-- Observable evidence for the real sequence: trial state, full/input frame,
the selected grid from that state, safety for every trial configuration, and
finite native execution of all trials followed by the final grid. -/
def TuningExecution {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (r : Request)
    (prepared : State α) (t : TensorAPI r prepared) (trials : List Config) (final : State α) : Prop :=
  ∃ after, runTrials M D r trials prepared=some after ∧
    OutputFrame r prepared after ∧
    (∀ o, after.mem r.A o=prepared.mem r.A o) ∧
    (∀ o, after.mem r.B o=prepared.mem r.B o) ∧
    host M D r after=.ok (r.C,final) ∧
    (∀ config∈trials, ∃ tc : TensorAPI (withConfig r config) prepared,
      Native.FullAccessSafety (withConfig r config) prepared tc) ∧
    (∃ native, Native.runConfigs M D r (Native.bases t) (trials++[r.config]) prepared=some native ∧
      native.mem=final.mem)

theorem tuning_execution {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (r : Request)
    (prepared after final : State α) (t : TensorAPI r prepared) (trials : List Config)
    (legal : ∀ config∈trials, config∈configurations r.target)
    (abi : ∀ config∈trials, programs (withConfig r config)<2^31)
    (hgrid : programs r<2^31)
    (doneTrials : runTrials M D r trials prepared=some after)
    (frame : OutputFrame r prepared after)
    (doneFinal : host M D r after=.ok (r.C,final)) :
    TuningExecution M D r prepared t trials final := by
  obtain ⟨hA,hB⟩ := frame.inputs r prepared after t.freshA t.freshB
  have gridFinal : runPrograms M D r (List.range (programs r)) after=some final := by
    obtain ⟨out,he⟩ := candidate_grid_executes M D r (List.range (programs r)) after
    have hh : out=final := by simpa [host,t.dimensions,t.contiguousA,he] using doneFinal
    simpa [hh] using he
  obtain ⟨native,typed,hn,ht,hm⟩ := Native.configs_correspondence M D r prepared t
    (trials++[r.config])
    (by intro c hc; rcases List.mem_append.mp hc with h|h
        · exact legal c h
        · have hc : c=r.config := by simpa using h
          simpa [hc] using t.selected)
    (by intro c hc; rcases List.mem_append.mp hc with h|h
        · exact abi c h
        · have hc : c=r.config := by simpa using h
          simpa [hc] using hgrid)
    prepared prepared rfl
  have typedFinal : typed=final := by
    symm
    simpa [runTrials_append,doneTrials,runTrials,gridFinal] using ht
  refine ⟨after,doneTrials,frame,hA,hB,doneFinal,?_,native,hn,by simpa [typedFinal] using hm⟩
  intro config hc
  exact ⟨t.reconfigure config (legal config hc),
    Native.full_access_safety _ _ _ (abi config hc)⟩

#axiomsClean programs_frame
#axiomsClean trials_frame
#axiomsClean Native.configs_correspondence
#axiomsClean tuning_execution
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
