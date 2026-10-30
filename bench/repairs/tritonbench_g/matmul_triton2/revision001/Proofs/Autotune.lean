import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.LogicalInputs
namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Autotune
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.Repairs.MatmulTriton1.Revision001
open SharedContract TensorBinding
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

/-- Actual successful tuner trials may repeat configurations; a cache hit has
an empty trial list. The selected configuration is launched after all trials. -/
structure Plan where
  trials : List (Fin 8)
  selected : Fin 8

def Frame (x : Host.Arguments) (c : Host.Tensor) (initial final : State α) : Prop :=
  ∀ r o, (r≠c.region ∨ ∀ i : Host.Index x, o≠Host.address c i) → final.mem r o=initial.mem r o

def launch (P : Fin 8 → Core.Primitives α) (a b : TensorBinding.Tensor)
    (ra rb : RegionName) (c : Host.Tensor) (choice : Fin 8) (s : State α) :=
  FiniteSource.grid (P choice) (Host.parameters (InputAdapter.arguments a b ra rb choice) c) s

def trials (P : Fin 8 → Core.Primitives α) (a b : TensorBinding.Tensor)
    (ra rb : RegionName) (c : Host.Tensor) : List (Fin 8) → State α → Option (State α)
  | [],s => some s
  | choice::rest,s => (launch P a b ra rb c choice s).bind (trials P a b ra rb c rest)

def call (P : Fin 8 → Core.Primitives α) (allocate : Host.Allocator α)
    (a b : TensorBinding.Tensor) (ra rb : RegionName) (plan : Plan) (s : State α) :=
  if SharedContract.checkTensorInputs Bool.false Bool.true a.metadata b.metadata=.ok () then do
    let (c,allocated) ← allocate (InputAdapter.arguments a b ra rb plan.selected) s
    if Host.Empty (InputAdapter.arguments a b ra rb plan.selected) then some (c,allocated)
    else do
      let warmed ← trials P a b ra rb c plan.trials allocated
      let final ← launch P a b ra rb c plan.selected warmed
      some (c,final)
  else none

def Launchable (allocate : Host.Allocator α) (a b : TensorBinding.Tensor)
    (ra rb : RegionName) (plan : Plan) (s : State α) : Prop :=
  ∀ c after, allocate (InputAdapter.arguments a b ra rb plan.selected) s=some (c,after) →
    ¬Host.Empty (InputAdapter.arguments a b ra rb plan.selected) →
    ∀ choice, choice∈plan.trials ∨ choice=plan.selected →
      FiniteIndex.count (Host.parameters (InputAdapter.arguments a b ra rb choice) c)<2^31

theorem allocated_choice (a b : TensorBinding.Tensor) (ra rb : RegionName)
    (selected choice : Fin 8) (initial allocated : State α) (c : Host.Tensor)
    (h : Host.Allocated (InputAdapter.arguments a b ra rb selected) initial allocated c) :
    Host.Allocated (InputAdapter.arguments a b ra rb choice) initial allocated c := by
  cases h
  constructor <;> assumption

theorem trial_frame (rest : Algebra α) (P : Fin 8 → Core.Primitives α)
    (a b : TensorBinding.Tensor) (la : Laws a) (lb : Laws b) (ra rb : RegionName)
    (h : SharedContract.InputContract Bool.false Bool.true a.metadata b.metadata)
    (selected choice : Fin 8) (initial allocated current : State α) (c : Host.Tensor)
    (allocation : Host.Allocated (InputAdapter.arguments a b ra rb selected) initial allocated c)
    (abi : InputAdapter.OutputABI c)
    (ready : FiniteIndex.count (Host.parameters (InputAdapter.arguments a b ra rb choice) c)<2^31) :
    ∃ final, launch P a b ra rb c choice current=some final ∧
      Frame (InputAdapter.arguments a b ra rb selected) c current final := by
  have ac := allocated_choice a b ra rb selected choice initial allocated c allocation
  have valid := InputAdapter.valid a b ra rb choice h
  obtain ⟨final,he,_,hf⟩ := FiniteSource.grid_correct rest (P choice)
    (Host.parameters (InputAdapter.arguments a b ra rb choice) c)
    (InputAdapter.canonical a b ra rb choice initial allocated c ac abi)
    (InputAdapter.ready a b ra rb choice h c ready)
    (InputAdapter.storage a b la lb ra rb choice h initial allocated c ac abi)
    (Host.geometry_valid _ valid initial allocated c ac) ac.freshA ac.freshB current
  refine ⟨final,he,?_⟩
  intro r o outside
  exact hf r o (outside.imp id (fun hi i _ => hi i))

theorem trials_frame (rest : Algebra α) (P : Fin 8 → Core.Primitives α)
    (a b : TensorBinding.Tensor) (la : Laws a) (lb : Laws b) (ra rb : RegionName)
    (h : SharedContract.InputContract Bool.false Bool.true a.metadata b.metadata)
    (selected : Fin 8) (initial allocated : State α) (c : Host.Tensor)
    (allocation : Host.Allocated (InputAdapter.arguments a b ra rb selected) initial allocated c)
    (abi : InputAdapter.OutputABI c) (choices : List (Fin 8))
    (ready : ∀ choice∈choices, FiniteIndex.count (Host.parameters (InputAdapter.arguments a b ra rb choice) c)<2^31)
    (current : State α) :
    ∃ final, trials P a b ra rb c choices current=some final ∧
      Frame (InputAdapter.arguments a b ra rb selected) c current final := by
  induction choices generalizing current with
  | nil => exact ⟨current,rfl,fun _ _ _ => rfl⟩
  | cons choice choices ih =>
    obtain ⟨next,he,hf⟩ := trial_frame rest P a b la lb ra rb h selected choice
      initial allocated current c allocation abi (ready choice (by simp))
    obtain ⟨final,hte,htf⟩ := ih (fun ch member => ready ch (by simp [member])) next
    refine ⟨final,?_,?_⟩
    · simp [trials,he,hte]
    · intro r o outside
      exact (htf r o outside).trans (hf r o outside)

/-- Every autotune trial is executed, then the selected kernel overwrites the
same fresh output. Its precision-specific result reads the original inputs;
trial arithmetic is never assumed equal to selected arithmetic. -/
theorem correct (rest : Algebra α) (P : Fin 8 → Core.Primitives α) (allocate : Host.Allocator α)
    (a b : TensorBinding.Tensor) (la : Laws a) (lb : Laws b) (ra rb : RegionName) (plan : Plan)
    (h : SharedContract.InputContract Bool.false Bool.true a.metadata b.metadata) (initial : State α)
    (allocation : CheckedHost.Allocation allocate (InputAdapter.arguments a b ra rb plan.selected) initial)
    (ready : Launchable allocate a b ra rb plan initial) :
    ∃ c final, call P allocate a b ra rb plan initial=some (c,final) ∧
      Host.Post (P plan.selected) (InputAdapter.arguments a b ra rb plan.selected) initial final c := by
  obtain ⟨c,allocated,ha,ac,abi⟩ := allocation
  have valid := InputAdapter.valid a b ra rb plan.selected h
  have checked := (SharedContract.checkTensorInputs_iff Bool.false Bool.true a.metadata b.metadata).mpr h
  by_cases empty : Host.Empty (InputAdapter.arguments a b ra rb plan.selected)
  · refine ⟨c,allocated,?_,ac.rank,ac.rows,ac.columns,ac.dtype,ac.contiguous,ac.device,ac.cuda,ac.resolved,
      ac.freshObjectA,ac.freshObjectB,?_,?_⟩
    · simp [call,checked,ha,empty]
    · intro i
      rcases empty with hm | hn
      · have := i.1.isLt; omega
      · have := i.2.isLt; omega
    · intro r o _
      exact congrFun (congrFun ac.memory r) o
  · obtain ⟨warmed,hw,fw⟩ := trials_frame rest P a b la lb ra rb h plan.selected initial allocated c ac abi
      plan.trials (fun choice member => ready c allocated ha empty choice (Or.inl member)) allocated
    obtain ⟨final,hfinal,values,frame⟩ := FiniteSource.grid_correct rest (P plan.selected)
      (Host.parameters (InputAdapter.arguments a b ra rb plan.selected) c)
      (InputAdapter.canonical a b ra rb plan.selected initial allocated c ac abi)
      (InputAdapter.ready a b ra rb plan.selected h c (ready c allocated ha empty _ (Or.inr rfl)))
      (InputAdapter.storage a b la lb ra rb plan.selected h initial allocated c ac abi)
      (Host.geometry_valid _ valid initial allocated c ac) ac.freshA ac.freshB warmed
    have outside : GridExecution.Outside c.region initial warmed := by
      intro r o other
      exact (fw r o (Or.inl other)).trans (congrFun (congrFun ac.memory r) o)
    refine ⟨c,final,?_,ac.rank,ac.rows,ac.columns,ac.dtype,ac.contiguous,ac.device,ac.cuda,ac.resolved,
      ac.freshObjectA,ac.freshObjectB,?_,?_⟩
    · simp [call,checked,ha,empty,hw,launch,hfinal]
    · intro i
      have value := values i trivial
      dsimp only at value
      rw [Core.logical_preserved (P plan.selected)
        (Host.parameters (InputAdapter.arguments a b ra rb plan.selected) c)
        initial warmed ac.freshA ac.freshB outside] at value
      exact value
    · intro r o other
      exact (frame r o (other.imp id (fun hi i _ => hi i))).trans
        ((fw r o other).trans (congrFun (congrFun ac.memory r) o))

#axiomsClean trial_frame
#axiomsClean trials_frame
#axiomsClean correct
end
end VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Autotune
