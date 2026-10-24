import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.InputAdapter
namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001.CheckedHost
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton1.Revision001
open SharedContract TensorBinding
noncomputable section
variable {α : Type} [Inhabited α]

def call (P : Core.Primitives α) (allocate : Host.Allocator α)
    (a b : TensorBinding.Tensor) (ra rb : RegionName) (choice : Fin 8) (s : State α) :=
  if SharedContract.checkTensorInputs Bool.false Bool.true a.metadata b.metadata=.ok () then
    FiniteHost.call P allocate (InputAdapter.arguments a b ra rb choice) s
  else none

/-- Pointwise successful allocation, including its local frame and capacity.
It does not require every admitted input to fit device memory. -/
def Allocation (allocate : Host.Allocator α) (x : Host.Arguments) (s : State α) : Prop :=
  ∃ c after, allocate x s=some (c,after) ∧ Host.Allocated x s after c ∧ InputAdapter.OutputABI c

theorem allocation_result (allocate : Host.Allocator α) (x : Host.Arguments) (s : State α)
    (law : Allocation allocate x s) (c : Host.Tensor) (after : State α)
    (result : allocate x s=some (c,after)) : Host.Allocated x s after c ∧ InputAdapter.OutputABI c := by
  obtain ⟨original,before,he,ha,hb⟩ := law
  have same := Option.some.inj (he.symm.trans result)
  rcases Prod.mk.inj same with ⟨rfl,rfl⟩
  exact ⟨ha,hb⟩

/-- Successful native launch supplies a representable grid. It is a local
runtime boundary for this actual allocation, not a Python input-size guard. -/
def Launchable (allocate : Host.Allocator α) (x : Host.Arguments) (s : State α) : Prop :=
  ∀ c after, allocate x s=some (c,after) → ¬Host.Empty x →
    FiniteIndex.count (Host.parameters x c)<2^31

theorem correct (rest : Algebra α) (P : Core.Primitives α) (allocate : Host.Allocator α)
    (a b : TensorBinding.Tensor) (la : Laws a) (lb : Laws b) (ra rb : RegionName) (choice : Fin 8)
    (h : SharedContract.InputContract Bool.false Bool.true a.metadata b.metadata) (initial : State α)
    (allocation : Allocation allocate (InputAdapter.arguments a b ra rb choice) initial)
    (launch : Launchable allocate (InputAdapter.arguments a b ra rb choice) initial) :
    ∃ c final, call P allocate a b ra rb choice initial=some (c,final) ∧
      Host.Post P (InputAdapter.arguments a b ra rb choice) initial final c := by
  have localLaw := allocation_result allocate (InputAdapter.arguments a b ra rb choice) initial allocation
  have allocated : ∃ c after, allocate (InputAdapter.arguments a b ra rb choice) initial=some (c,after) ∧
      Host.Allocated (InputAdapter.arguments a b ra rb choice) initial after c := by
    obtain ⟨c,after,he,hc,_⟩ := allocation
    exact ⟨c,after,he,hc⟩
  have result := FiniteHost.correct rest P allocate (InputAdapter.arguments a b ra rb choice)
    (InputAdapter.valid a b ra rb choice h) initial allocated
    (fun c after result => InputAdapter.canonical a b ra rb choice initial after c
      (localLaw c after result).1 (localLaw c after result).2)
    (fun c after result nonempty => InputAdapter.ready a b ra rb choice h c
      (launch c after result nonempty))
    (fun c after result => InputAdapter.storage a b la lb ra rb choice h initial after c
      (localLaw c after result).1 (localLaw c after result).2)
  simpa only [call,if_pos ((SharedContract.checkTensorInputs_iff Bool.false Bool.true a.metadata b.metadata).mpr h)] using result

theorem rejected (P : Core.Primitives α) (allocate : Host.Allocator α)
    (a b : TensorBinding.Tensor) (ra rb : RegionName) (choice : Fin 8) (s : State α)
    (h : ¬SharedContract.InputContract Bool.false Bool.true a.metadata b.metadata) :
    call P allocate a b ra rb choice s=none := by
  simp [call,(SharedContract.checkTensorInputs_iff Bool.false Bool.true a.metadata b.metadata).not.mpr h]

#axiomsClean correct
#axiomsClean rejected
end
end VeriTile.Bench.Repairs.MatmulTriton2.Revision001.CheckedHost
