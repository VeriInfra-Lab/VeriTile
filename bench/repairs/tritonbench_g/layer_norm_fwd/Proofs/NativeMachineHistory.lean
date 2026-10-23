import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeMachineGrid

/-! Actual predecessor-state execution of benchmark and selected launches.
The dispatch controller supplies only profiles and setup outcomes. Every
successful launch computes its own output with the checked native machine. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow Address PhysicalDomain NativeABI
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

structure Schedule (block : Nat) where
  reductions : Nat → Whole.Reductions block
  plan : AffinePlan
  plan1 : AffinePlan

def Config.withSchedule {α : Type} (c : Config α block) (schedule : Schedule block) : Config α block :=
  {c with reductions:=schedule.reductions,plan:=schedule.plan,plan1:=schedule.plan1}

inductive DispatchEvent (block : Nat) where
  | launch (schedule : Schedule block)
  | setupFailure (code : Nat)

structure DispatchResult {α : Type} (c : Config α block) where
  memory : Memory c
  error : Option Nat
  completed : List (Schedule block)

def history {α : Type} [Inhabited α] (c : Config α block) :
    List (DispatchEvent block) → Memory c → DispatchResult c
  | [],physical => ⟨physical,none,[]⟩
  | .setupFailure code::_,physical => ⟨physical,some code,[]⟩
  | .launch schedule::rest,physical =>
    let next := executeGrid (c.withSchedule schedule) physical
    let result := history c rest next
    ⟨result.memory,result.error,schedule::result.completed⟩

theorem history_failure {α : Type} [Inhabited α] (c : Config α block) (code : Nat)
    (rest : List (DispatchEvent block)) (physical : Memory c) :
    history c (.setupFailure code::rest) physical=⟨physical,some code,[]⟩ := rfl

theorem history_frame {α : Type} [Inhabited α] (c : Config α block)
    (events : List (DispatchEvent block)) (physical : Memory c)
    (i : Fin (layout c.allocations).capacity) (hi : ∀ row : Fin c.M, Outside c row i) :
    (history c events physical).memory i=physical i := by
  induction events generalizing physical with
  | nil => rfl
  | cons event rest ih =>
    cases event with
    | setupFailure code => rfl
    | launch schedule =>
      change (history c rest (executeGrid (c.withSchedule schedule) physical)).memory i=physical i
      rw [ih]
      exact grid_frame (c.withSchedule schedule) physical i hi

theorem grid_reads {α : Type} [Inhabited α] (c : Config α block)
    (hsep : Domain.ReadWriteSeparated c.flags c.regions)
    (hgeo : NativeGeometry.Geometry c.flags c.regions c.strides c.M c.N) (physical : Memory c) (template : State α) :
    TypedSource.ReadsEqual c.flags c.regions (view c (executeGrid c physical) template) (view c physical template) := by
  rw [grid_correspondence c hgeo physical template]
  have hp := (NativeSchedule.execute_grid_post c.ops c.reductions c.plan c.plan1 c.flags c.regions c.strides
    c.M c.N c.eps c.p c.accepted.blockPositive (view c physical template) hsep hgeo).2
  exact NativeSchedule.grid_input_preserved c.ops c.reductions c.plan c.plan1 c.flags c.regions c.strides
    c.M c.N c.eps c.p (view c physical template) _ hsep hp

/-- All benchmark launches preserve every enabled input view, even on the
path ending in a later setup failure. This is derived from machine execution,
not required as a dispatch API postcondition. -/
theorem history_reads {α : Type} [Inhabited α] (c : Config α block)
    (hsep : Domain.ReadWriteSeparated c.flags c.regions)
    (hgeo : NativeGeometry.Geometry c.flags c.regions c.strides c.M c.N) (events : List (DispatchEvent block))
    (physical : Memory c) (template : State α) :
    TypedSource.ReadsEqual c.flags c.regions (view c (history c events physical).memory template)
      (view c physical template) := by
  induction events generalizing physical with
  | nil => intro _ _ _; rfl
  | cons event rest ih =>
    cases event with
    | setupFailure code => intro _ _ _; rfl
    | launch schedule =>
      intro region hr offset
      change (view c (history c rest (executeGrid (c.withSchedule schedule) physical)).memory template).mem region offset=_
      rw [ih _ region hr offset]
      exact grid_reads (c.withSchedule schedule) hsep hgeo physical template region hr offset

/-- Successful benchmark prefixes execute the suffix from their actual last
physical state. A failed prefix stops without executing any suffix launch. -/
theorem history_append {α : Type} [Inhabited α] (c : Config α block)
    (beforeEvents suffix : List (DispatchEvent block)) (physical : Memory c) :
    history c (beforeEvents++suffix) physical=
      match (history c beforeEvents physical).error with
      | some _ => history c beforeEvents physical
      | none => let after := history c suffix (history c beforeEvents physical).memory
                ⟨after.memory,after.error,(history c beforeEvents physical).completed++after.completed⟩ := by
  induction beforeEvents generalizing physical with
  | nil => rfl
  | cons event rest ih =>
    cases event with
    | setupFailure code => rfl
    | launch schedule =>
      simp only [List.cons_append,history]
      rw [ih]
      cases he : (history c rest (executeGrid (c.withSchedule schedule) physical)).error <;> simp [he]

#axiomsClean history_failure
#axiomsClean history_frame
#axiomsClean grid_reads
#axiomsClean history_reads
#axiomsClean history_append
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
