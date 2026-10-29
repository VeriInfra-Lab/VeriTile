import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.HostSource

namespace VeriTile.Bench.Repairs.FastCeLoss.Host
open Source API HostStages
open CrossEntropy1.Host (Step)
noncomputable section
set_option maxHeartbeats 4000000

theorem bind_success {α σ ρ τ : Type} {first : Action α σ ρ} {next : ρ → Action α σ τ}
    {initial final : World α σ} {result : τ}
    (success : bindAction first next initial=.ok result final) :
    ∃ value middle,first initial=.ok value middle ∧ next value middle=.ok result final := by
  cases h : first initial with
  | error message failed => simp only [bindAction,h] at success; cases success
  | ok value middle => exact ⟨value,middle,rfl,by simpa only [bindAction,h] using success⟩

theorem allocate_memory {α σ : Type} (runtime : Runtime σ) (shape : List Nat) (dtype : DType) (device : Nat)
    (initial final : World α σ) (output : Descriptor)
    (success : allocate runtime shape dtype device initial=.ok output final) :
    final.memory=initial.memory ∧ final.integer=initial.integer ∧ final.calls=initial.calls ∧
      runtime.allocate shape dtype device initial.native=.ok output final.native := by
  cases h : runtime.allocate shape dtype device initial.native with
  | error message native => simp only [allocate,h] at success; cases success
  | ok value native =>
    simp only [allocate,h,Step.ok.injEq] at success
    rcases success with ⟨rfl,rfl⟩
    exact ⟨rfl,rfl,rfl,rfl⟩

theorem view_memory {α σ : Type} (runtime : Runtime σ) (input : Descriptor) (shape : List Nat)
    (initial final : World α σ) (output : Descriptor)
    (success : view runtime input shape initial=.ok output final) :
    final.memory=initial.memory ∧ final.integer=initial.integer ∧ final.calls=initial.calls ∧
      runtime.view input shape initial.native=.ok output final.native := by
  cases h : runtime.view input shape initial.native with
  | error message native => simp only [view,h] at success; cases success
  | ok value native =>
    simp only [view,h,Step.ok.injEq] at success
    rcases success with ⟨rfl,rfl⟩
    exact ⟨rfl,rfl,rfl,rfl⟩

theorem expand_memory {α σ : Type} (runtime : Runtime σ) (input : Descriptor) (rows : Nat)
    (initial final : World α σ) (output : Descriptor)
    (success : expand runtime input rows initial=.ok output final) :
    final.memory=initial.memory ∧ final.integer=initial.integer ∧ final.calls=initial.calls ∧
      runtime.expand input rows initial.native=.ok output final.native := by
  cases h : runtime.expand input rows initial.native with
  | error message native => simp only [expand,h] at success; cases success
  | ok value native =>
    simp only [expand,h,Step.ok.injEq] at success
    rcases success with ⟨rfl,rfl⟩
    exact ⟨rfl,rfl,rfl,rfl⟩

theorem perform_memory {α σ : Type} (runtime : Runtime σ) (request : Call)
    (floats : Memory α → Memory α) (integers : (Address → Int) → Address → Int)
    (initial final : World α σ) (result : Unit)
    (success : perform runtime request floats integers initial=.ok result final) :
    final.memory=floats initial.memory ∧ final.integer=integers initial.integer ∧
      final.calls=initial.calls++[request] ∧ runtime.perform request initial.native=.ok () final.native := by
  cases h : runtime.perform request initial.native with
  | error message native => simp only [perform,h] at success; cases success
  | ok value native =>
    cases value
    simp only [perform,h,Step.ok.injEq] at success
    rcases success with ⟨_,rfl⟩
    exact ⟨rfl,rfl,rfl,rfl⟩

structure BackwardExecution {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (runtime : Runtime σ) (stream : Nat) (saved : ForwardReturn) (upstream : Descriptor)
    (initial final : World α σ) (result : WholeReturn) where
  divided : Descriptor
  allocated : World α σ
  allocation : allocate runtime [] .fp32 saved.scalar.device initial=.ok divided allocated
  dividedState : World α σ
  division : perform runtime (call stream .divideBackward ([upstream]++saved.count.toList)
    [divided] saved.args.items.toList)
    (fun memory => writeRows divided.tensor 1 (fun _ => a.upstreamDiv (a.load (memory (addr upstream.tensor 0)))
      (denominatorValue a scalars saved.args saved.count initial)) memory) id allocated=.ok () dividedState
  expanded : Descriptor
  expandedState : World α σ
  expansion : expand runtime divided saved.args.shape.rows dividedState=.ok expanded expandedState
  prepared : World α σ
  expansionEvent : perform runtime (call stream .expandSumBackward [divided] [expanded]) id id expandedState=.ok () prepared
  computed : World α σ
  kernel : perform runtime (call stream (.launch .backward [saved.args.shape.rows,saved.args.shape.blocks]
    4096 8 saved.context.options) [saved.context.logits,expanded,saved.context.lse,saved.context.labels]
    [saved.context.logits] [saved.args.capArgument,saved.args.scaleArgument]
    [saved.context.logits.strides.headD 0,expanded.strides.headD 0,(saved.args.shape.vocab : Int)])
    (backwardGrid a saved.context.options saved.args.shape (returnedBuffers saved expanded)
      (labelValues saved.context.labels prepared)) id prepared=.ok () computed
  gradient : Descriptor
  viewed : World α σ
  pullback : view runtime saved.context.logits [saved.args.batch,saved.args.sequence,saved.args.shape.vocab]
    computed=.ok gradient viewed
  returned : perform runtime (call stream .returnValue [saved.context.logits,gradient] []) id id viewed=.ok () final
  resultExact : result=⟨saved,[some saved.context.logits.tensor,none,none,none],gradient,expanded⟩

theorem backward_success {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (runtime : Runtime σ) (stream : Nat) (saved : ForwardReturn) (upstream : Descriptor)
    (initial final : World α σ) (result : WholeReturn)
    (success : backward a scalars runtime stream saved upstream initial=.ok result final) :
    Nonempty (BackwardExecution a scalars runtime stream saved upstream initial final result) := by
  unfold backward at success
  obtain ⟨readInitial,initialState,hread,rest⟩ := bind_success (first:=read) success
  simp only [read,Step.ok.injEq] at hread
  rcases hread with ⟨rfl,rfl⟩
  obtain ⟨divided,allocated,halloc,rest⟩ := bind_success rest
  obtain ⟨unit,dividedState,hdivide,rest⟩ := bind_success rest
  cases unit
  obtain ⟨expanded,expandedState,hexpand,rest⟩ := bind_success rest
  obtain ⟨unit,prepared,hexpansionEvent,rest⟩ := bind_success rest
  cases unit
  obtain ⟨readPrepared,preparedState,hreadPrepared,rest⟩ := bind_success rest
  simp only [read,Step.ok.injEq] at hreadPrepared
  rcases hreadPrepared with ⟨rfl,rfl⟩
  obtain ⟨unit,computed,hkernel,rest⟩ := bind_success rest
  cases unit
  obtain ⟨gradient,viewed,hview,rest⟩ := bind_success rest
  obtain ⟨unit,returned,hreturn,rest⟩ := bind_success rest
  cases unit
  change Step.ok _ returned=Step.ok result final at rest
  rcases Step.ok.inj rest with ⟨rfl,rfl⟩
  exact ⟨⟨divided,allocated,halloc,dividedState,hdivide,expanded,expandedState,hexpand,
    prepared,hexpansionEvent,computed,hkernel,gradient,viewed,hview,hreturn,rfl⟩⟩

theorem whole_success {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (runtime : Runtime σ) (stream : Nat) (args : Arguments) (upstream : Descriptor)
    (initial final : World α σ) (result : WholeReturn)
    (success : whole a scalars runtime stream args upstream initial=.ok result final) :
    ∃ saved middle, forward a scalars runtime stream args initial=.ok saved middle ∧
      backward a scalars runtime stream saved upstream middle=.ok result final := bind_success success

#axiomsClean allocate_memory
#axiomsClean perform_memory
#axiomsClean backward_success
#axiomsClean whole_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Host
