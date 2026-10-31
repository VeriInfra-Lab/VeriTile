import bench.tritonbench_g.attention_score.Proofs.Reused.FPOutput
import bench.tritonbench_g.attention_score.Proofs.Reused.WholeGrid

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPGrid
open VeriTile Triton FP.Structural FP.ProfiledRewrite Geometry FPEngine
open WholeGrid (Job jobs)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def programState (c : Shape) (s : State α) (job : Job c) : State α :=
  {s with pids := fun dim => if dim=0 then job.2.2.val else if dim=1 then job.1.val*c.H+job.2.1.val else 0}

def invoke (M : Algebra α) (dot : Dot α) (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (job : Job c) (s : State α) : Option (State α) :=
  FP.Structural.exec (engine (withDot M dot))
    (FPEngine.kernel c.B c.D (parameters c offset size scale window complement)) (programState c s job)

def run (M : Algebra α) (dot : Dot α) (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) : List (Job c) → State α → Option (State α)
  | [],s => some s
  | job::rest,s => (invoke M dot c offset size scale window complement job s).bind
      (run M dot c offset size scale window complement rest)

def InputsSame (initial current : State α) : Prop :=
  ∀ (r : RegionName) a, r≠"out" → current.mem r a=initial.mem r a

def values (M : Algebra α) (dot : Dot α) (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (s : State α) (job : Job c) (i : Fin c.B) : α :=
  M.cast (some .fp32) .real .real
    (FPLoop.accumulated M dot c.B c.D (parameters c offset size scale window complement) (programState c s job) i)

theorem values_same (M : Algebra α) (dot : Dot α) (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (initial current : State α) (same : InputsSame initial current) (job : Job c) :
    values M dot c offset size scale window complement current job =
      values M dot c offset size scale window complement initial job := by
  have hq : ∀ a, current.mem "q" a=initial.mem "q" a := fun a => same _ a (by decide)
  have hk : ∀ a, current.mem "k" a=initial.mem "k" a := fun a => same _ a (by decide)
  have hm : ∀ a, current.mem "m" a=initial.mem "m" a := fun a => same _ a (by decide)
  have hu : FPValues.update M dot c.B c.D (parameters c offset size scale window complement) (programState c current job)=
      FPValues.update M dot c.B c.D (parameters c offset size scale window complement) (programState c initial job) := by
    funext row acc col
    delta FPValues.update FPValues.probability FPValues.raw FPValues.query FPValues.normalizer
      FPValues.distance FPContext.key FPContext.keyStart FPContext.baseQ FPContext.baseK FPContext.baseM
    simp only [programState,parameters,hq,hk,hm]
  funext i
  simp only [values,FPLoop.accumulated,hu]

theorem output_address (c : Shape) (s : State α) (job : Job c)
    (offset size : Int) (scale : ℝ) (window complement : Bool) (i : Fin c.B) :
    FPOutput.address c.B (parameters c offset size scale window complement) (programState c s job) (i,())=
      WholeGrid.address c job i := by
  obtain ⟨hz,hh⟩ := DenseSource.batch_head c job.1.val job.2.1
  simp [FPOutput.address,parameters,programState,hz,hh,WholeGrid.address,Shape.outputBase,
    Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]

def CorrectAt (M : Algebra α) (dot : Dot α) (c : Shape) (initial current : State α)
    (offset size : Int) (scale : ℝ) (window complement : Bool) (job : Job c) : Prop :=
  ∀ i : Fin c.B, WholeGrid.active c job i →
    current.mem "out" (WholeGrid.address c job i)=.mk .real (values M dot c offset size scale window complement initial job i)

theorem step (M : Algebra α) (dot : Dot α) (c : Shape) (initial current : State α)
    (same : InputsSame initial current) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (job : Job c) :
    ∃ t, invoke M dot c offset size scale window complement job current=some t ∧ InputsSame initial t ∧
      CorrectAt M dot c initial t offset size scale window complement job ∧
      (∀ old, old≠job → CorrectAt M dot c initial current offset size scale window complement old →
        CorrectAt M dot c initial t offset size scale window complement old) := by
  let p := parameters c offset size scale window complement
  let s := programState c current job
  let a := FPLoop.accumulated M dot c.B c.D p s
  obtain ⟨t,ht,hmem⟩ := FPOutput.source_memory M dot c.B c.D p s
  refine ⟨t,ht,?_,?_,?_⟩
  · intro r off hr
    rw [hmem]
    exact (FPOutput.frame M c.B p s a r off (Or.inl hr)).trans (same r off hr)
  · intro i hi
    have hr := FPOutput.readback M c.B p s a (i,()) hi
    rw [output_address] at hr
    rw [hmem]
    exact hr.trans (congrArg (Cell.mk .real) (congrFun
      (values_same M dot c offset size scale window complement initial current same job) i))
  · intro old hne hold i hi
    have hf := FPOutput.frame M c.B p s a "out" (WholeGrid.address c old i) (Or.inr (by
      intro j hj he
      rcases j with ⟨j,⟨⟩⟩
      have he' : WholeGrid.address c old i=WholeGrid.address c job j := by
        exact he.trans (output_address c current job offset size scale window complement j)
      exact hne (WholeGrid.address_owner c old job i j hi hj he')))
    rw [hmem,hf]
    exact hold i hi

theorem run_correct (M : Algebra α) (dot : Dot α) (c : Shape) (initial current : State α)
    (same : InputsSame initial current) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (order done : List (Job c))
    (correct : ∀ job∈done, CorrectAt M dot c initial current offset size scale window complement job) :
    ∃ t, run M dot c offset size scale window complement order current=some t ∧ InputsSame initial t ∧
      ∀ job∈done++order, CorrectAt M dot c initial t offset size scale window complement job := by
  induction order generalizing current done with
  | nil => exact ⟨current,rfl,same,by simpa using correct⟩
  | cons job rest ih =>
      obtain ⟨next,hn,hs,hnew,hkeep⟩ := step M dot c initial current same offset size scale window complement job
      have hc : ∀ old∈done++[job], CorrectAt M dot c initial next offset size scale window complement old := by
        intro old ho
        by_cases he : old=job
        · simpa [he] using hnew
        · apply hkeep old he
          exact correct old (by simpa [he] using ho)
      obtain ⟨t,ht,hi,hout⟩ := ih next hs (done++[job]) hc
      refine ⟨t,?_,hi,?_⟩
      · simp only [run,hn,Option.bind_some]; exact ht
      · simpa only [List.append_assoc,List.singleton_append] using hout

theorem whole_grid (M : Algebra α) (dot : Dot α) (c : Shape) (initial : State α)
    (offset size : Int) (scale : ℝ) (window complement : Bool) :
    ∃ t, run M dot c offset size scale window complement (jobs c) initial=some t ∧ InputsSame initial t ∧
      ∀ job, CorrectAt M dot c initial t offset size scale window complement job := by
  obtain ⟨t,ht,hs,ho⟩ := run_correct M dot c initial initial (by intro _ _ _; rfl)
    offset size scale window complement (jobs c) [] (by simp)
  exact ⟨t,ht,hs,fun job => ho job (by simp [jobs])⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPGrid
