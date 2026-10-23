import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostPrepared

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.HostControl
open VeriTile.Triton FP.Structural HostSnapshots HostPrepared
noncomputable section

structure API (α : Host.Role → Type) where
  enter : Except Unit Unit
  snapshots : HostSnapshots.API α
  lists : ToListAPI
  launch : Nat → Except Unit Unit
  leave : Except Unit Unit

structure PreparedCall {α : Host.Role → Type} (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View α role) (caller : List Window) where
  snapshots : Prepared entry caller
  lists : Lists snapshots
  segments : List Segment
  validated : validateSegments shape (raw batches lists) = some segments

/-- After successful source shape checks, context entry precedes all six
snapshots and three tolist calls. Validation completes before a launch exists. -/
def prepareInside {α : Host.Role → Type} (api : API α) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View α role) (caller : List Window) :
    Except Host.Stage (PreparedCall shape batches entry caller) := do
  let snapshots ← HostSnapshots.prepare api.snapshots entry caller
  let lists ← toLists api.lists snapshots
  match h : validateSegments shape (raw batches lists) with
  | none => .error .validate
  | some segments => pure ⟨snapshots,lists,segments,h⟩

/-- Python's context manager exits even when an operation inside raises.
An exit failure replaces the pending exception; successful preparation keeps
the context entered until the private launches have finished. -/
def prepare {α : Host.Role → Type} (api : API α) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View α role) (caller : List Window) :
    Except Host.Stage (PreparedCall shape batches entry caller) := do
  let _ ← api.enter.mapError (fun _ => .enter)
  match prepareInside api shape batches entry caller with
  | .ok setup => .ok setup
  | .error stage => match api.leave with
    | .ok _ => .error stage
    | .error _ => .error .leave

/-- Runtime submission failures are local API failures; successful submissions
execute the actual source batch grid. Kernel failure is kept visible. -/
def launchFrom {α : Type} [Inhabited α] (ready : Nat → Except Unit Unit)
    (M : Algebra α) (abi : SourceABI) (args : KernelArgs) (columns : Nat) :
    Nat → List Segment → State α → Except Host.Stage (State α)
  | _,[],state => .ok state
  | batch,segment::segments,state =>
      if segment.active ∧ 0 < columns then do
        let _ ← (ready batch).mapError (fun _ => .launch batch)
        let next ← match runSourceJobs M abi args (batchJobs columns batch segment) state with
          | none => .error (.launch batch)
          | some final => .ok final
        launchFrom ready M abi args columns (batch+1) segments next
      else launchFrom ready M abi args columns (batch+1) segments state

theorem launchFrom_sound {α : Type} [Inhabited α] (ready : Nat → Except Unit Unit)
    (M : Algebra α) (abi : SourceABI) (args : KernelArgs) (columns batch : Nat)
    (segments : List Segment) (state final : State α)
    (success : launchFrom ready M abi args columns batch segments state = .ok final) :
    runSourceJobs M abi args (jobsFrom columns batch segments) state = some final := by
  induction segments generalizing batch state with
  | nil => simpa [launchFrom,runSourceJobs,jobsFrom] using success
  | cons segment segments ih =>
      unfold launchFrom at success
      by_cases active : segment.active ∧ 0 < columns
      · rw [if_pos active] at success
        cases hready : ready batch with
        | error error =>
            simp only [hready,Except.mapError] at success
            change Except.error (Host.Stage.launch batch) = Except.ok final at success
            cases success
        | ok result =>
          cases hrun : runSourceJobs M abi args (batchJobs columns batch segment) state with
          | none =>
              simp only [hready,hrun,Except.mapError] at success
              change Except.error (Host.Stage.launch batch) = Except.ok final at success
              cases success
          | some next =>
              have hnext : launchFrom ready M abi args columns (batch+1) segments next = .ok final := by
                simpa [hready,hrun] using success
              rw [jobsFrom,source_jobs_append,hrun,Option.bind_some]
              exact ih _ _ hnext
      · rw [if_neg active] at success
        simpa [jobsFrom,batchJobs,active] using ih _ _ success

theorem launchFrom_complete {α : Type} [Inhabited α] (ready : Nat → Except Unit Unit)
    (M : Algebra α) (abi : SourceABI) (args : KernelArgs) (columns batch : Nat)
    (segments : List Segment) (state final : State α)
    (available : ∀ batch, ready batch = .ok ())
    (execution : runSourceJobs M abi args (jobsFrom columns batch segments) state = some final) :
    launchFrom ready M abi args columns batch segments state = .ok final := by
  induction segments generalizing batch state with
  | nil => simpa [launchFrom,runSourceJobs,jobsFrom] using execution
  | cons segment segments ih =>
      unfold launchFrom
      by_cases active : segment.active ∧ 0 < columns
      · rw [if_pos active]
        rw [jobsFrom,source_jobs_append] at execution
        cases hrun : runSourceJobs M abi args (batchJobs columns batch segment) state with
        | none => simp [hrun] at execution
        | some next =>
            have hexec : runSourceJobs M abi args (jobsFrom columns (batch+1) segments) next = some final := by
              simpa [hrun] using execution
            simpa [available,hrun] using ih (batch+1) next hexec
      · rw [if_neg active]
        apply ih
        simpa [jobsFrom,batchJobs,active] using execution

/-- A completed private phase includes successful context exit. The returned
state is only then available to the caller-copy phase. -/
def privatePhase {α : Type} [Inhabited α] (ready : Nat → Except Unit Unit) (leave : Except Unit Unit)
    (M : Algebra α) (abi : SourceABI) (args : KernelArgs) (shape : Shape)
    (segments : List Segment) (state : State α) : Except Host.Stage (State α) :=
  let launched := launchFrom ready M abi args shape.columns 0 segments state
  match leave with
  | .error _ => .error .leave
  | .ok _ => launched

theorem privatePhase_complete {α : Type} [Inhabited α] (ready : Nat → Except Unit Unit)
    (M : Algebra α) (abi : SourceABI) (args : KernelArgs) (shape : Shape)
    (segments : List Segment) (state final : State α)
    (available : ∀ batch, ready batch = .ok ())
    (execution : runSourceJobs M abi args (jobs shape.columns segments) state = some final) :
    privatePhase ready (.ok ()) M abi args shape segments state = .ok final := by
  have h := launchFrom_complete ready M abi args shape.columns 0 segments state final available execution
  unfold privatePhase
  rw [h]

#axiomsClean launchFrom_sound
#axiomsClean launchFrom_complete
#axiomsClean privatePhase_complete
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.HostControl
