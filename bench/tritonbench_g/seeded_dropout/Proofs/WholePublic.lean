import bench.tritonbench_g.seeded_dropout.Proofs.WholeIndex
import bench.tritonbench_g.seeded_dropout.Proofs.WholeMath

namespace VeriTile.Bench.TritonBenchG.SeededDropout.Whole
open VeriTile.Triton TypedSource FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

structure Runtime (α : Type) (input : Tensor) (entry : PhysicalMemory α) (preexisting : Nat → Prop) where
  allocate : Except String (Allocated α input entry preexisting)
  setup : (a : Allocated α input entry preexisting) → Except String (Schedule a)

inductive Outcome (α : Type) where
  | allocationFailure (reason : String)
  | assertionFailure (allocated : Tensor) (memory : PhysicalMemory α)
  | runtimeFailure (reason : String) (allocated : Tensor) (memory : PhysicalMemory α)
  | kernelFailure
  | returned (out : Tensor) (memory : PhysicalMemory α)

/-- The only numerical execution is the actual source body. Allocator and
launch setup cannot supply final outputs. Empty inputs select the empty grid. -/
def executeHost [Inhabited α] (M : Algebra α) (kind : Random.IntegerType) (p : ℝ)
    (seed : SeedCarrier kind) (rt : Runtime α input entry preexisting) : Outcome α :=
  match rt.allocate with
  | .error reason => .allocationFailure reason
  | .ok a =>
    if input.contiguous then
      match rt.setup a with
      | .error reason => .runtimeFailure reason a.out a.memory
      | .ok schedule =>
        match entryRun M a kind p seed schedule with
        | none => .kernelFailure
        | some final => .returned a.out (a.commit final)
    else .assertionFailure a.out a.memory

def ReturnedSpec [Inhabited α] (formula : Nat → α → α)
    (a : Allocated α input entry preexisting) (memory : PhysicalMemory α) : Prop :=
  a.out.shape = input.shape ∧ a.out.dtype = input.dtype ∧ a.out.cudaDevice = input.cudaDevice ∧
  a.out.contiguous = Bool.true ∧
  (∀ i, i < input.numel → a.out.read memory i = formula i (input.read entry i)) ∧
  (∀ allocation offset, PhysicalOutside a allocation offset →
    memory allocation offset = a.memory allocation offset) ∧
  (∀ allocation offset, preexisting allocation → memory allocation offset = entry allocation offset) ∧
  (∀ i, input.read memory i = input.read entry i)

def HostSpec [Inhabited α] (formula : Nat → α → α)
    (rt : Runtime α input entry preexisting) (outcome : Outcome α) : Prop :=
  (∃ reason, rt.allocate = .error reason ∧ outcome = .allocationFailure reason) ∨
  (∃ a, rt.allocate = .ok a ∧ input.contiguous = Bool.false ∧
    outcome = .assertionFailure a.out a.memory ∧
    ∀ allocation offset, preexisting allocation → a.memory allocation offset = entry allocation offset) ∨
  (∃ a reason, rt.allocate = .ok a ∧ input.contiguous = Bool.true ∧ rt.setup a = .error reason ∧
    outcome = .runtimeFailure reason a.out a.memory ∧
    ∀ allocation offset, preexisting allocation → a.memory allocation offset = entry allocation offset) ∨
  (∃ a schedule final, rt.allocate = .ok a ∧ input.contiguous = Bool.true ∧ rt.setup a = .ok schedule ∧
    outcome = .returned a.out (a.commit final) ∧ ReturnedSpec formula a (a.commit final))

theorem host_outcomes [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (compare : M.compareLt none .real = some lt) (kind : Random.IntegerType) (p : ℝ)
    (seed : SeedCarrier kind) (validSeed : kind.inRange (seedInt kind seed) = Bool.true)
    (rt : Runtime α input entry preexisting)
    (valid : ∀ a, rt.allocate = .ok a → input.contiguous = Bool.true → Domain a) :
    HostSpec (scalar M lt p (seedInt kind seed)) rt (executeHost M kind p seed rt) := by
  cases ha : rt.allocate with
  | error reason => exact Or.inl ⟨reason,ha,by simp [executeHost,ha]⟩
  | ok a =>
    cases hc : input.contiguous with
    | «false» => exact Or.inr (Or.inl ⟨a,ha,hc,by simp [executeHost,ha,hc],a.frame⟩)
    | «true» =>
      cases hs : rt.setup a with
      | error reason =>
        exact Or.inr (Or.inr (Or.inl ⟨a,reason,ha,hc,hs,by simp [executeHost,ha,hc,hs],a.frame⟩))
      | ok schedule =>
        obtain ⟨final,ran,values,frame,entryFrame⟩ := entry_result M lt compare a (valid a ha hc)
          hc kind p seed validSeed schedule
        apply Or.inr; apply Or.inr; apply Or.inr
        refine ⟨a,schedule,final,ha,hc,hs,?_,a.shape,a.dtype,a.device,a.contiguous hc,
          values,frame,entryFrame,?_⟩
        · simp [executeHost,ha,hc,hs,ran]
        · intro i
          exact entryFrame input.allocation (input.offset i) a.inputPreexisting

theorem real_host_outcomes (kind : Random.IntegerType) (p : ℝ)
    (seed : SeedCarrier kind) (validSeed : kind.inRange (seedInt kind seed) = Bool.true)
    (rt : Runtime ℝ input entry preexisting)
    (valid : ∀ a, rt.allocate = .ok a → input.contiguous = Bool.true → Domain a) :
    HostSpec (realFormula p (seedInt kind seed)) rt (executeHost realAlgebra kind p seed rt) := by
  have h := host_outcomes realAlgebra (fun a b => decide (a < b)) rfl kind p seed validSeed rt valid
  have equal : scalar realAlgebra (fun a b => decide (a < b)) p (seedInt kind seed) =
      realFormula p (seedInt kind seed) := by funext i x; exact scalar_real p _ i x
  simpa only [equal] using h

#guard_msgs (drop info) in
#specNonCircular realFormula avoiding [Layout.source,sourceRun,tileExec,runTiles,entryRun,executeHost]
#guard_msgs (drop info) in
#specNonCircular scalar avoiding [Layout.source,sourceRun,tileExec,runTiles,entryRun,executeHost]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SeededDropout.Whole
