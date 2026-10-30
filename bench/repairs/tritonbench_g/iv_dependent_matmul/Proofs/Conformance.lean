import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Execution
import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Wrapper

/-! Local bit-level primitive boundaries, never exact-Real equations.
The expected algebra is `Execution.fp cfg D F r`: fp32 Dot+Add or separate
DotAcc, terminal binary16 narrowing, and binary32 widening. Native SM80 builds
use the compiler-selected TF32 MMA callback; selecting that callback's semantics
and proving these local equalities remain numerical conformance obligations.
There is no claim that sequential software Dot equals native tensor-core Dot. -/
namespace VeriTile.Bench.Repairs.IvDependentMatmul.Conformance
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Kernels Whole
open Execution (Arithmetic Acc AValues BValues)
open TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 5000000

def limit (r : Request) := (Native.ceilWord r.inner r.blockInner).toNat

def accumulatorAt [Inhabited Word] (m : Arithmetic Word r) (g : Graph)
    (ba bb pid : Nat) (s : State Word) : Nat → Acc Word r
  | 0 => fun _ => m.zero
  | k+1 => Execution.step m g ba bb pid s k (accumulatorAt m g ba bb pid s k)

private theorem range_prefix [Inhabited Word] (m : Arithmetic Word r) (g : Graph)
    (ba bb pid : Nat) (s : State Word) (cur : Nat) (hc : cur≤limit r) :
    foldRange (Execution.step m g ba bb pid s) cur (limit r) 1
      (accumulatorAt m g ba bb pid s cur)=accumulatorAt m g ba bb pid s (limit r) := by
  by_cases hk : cur<limit r
  · rw [foldRange]
    simp only [Nat.one_ne_zero,hk,↓reduceDIte]
    exact range_prefix m g ba bb pid s (cur+1) (by omega)
  · have he : cur=limit r := by omega
    subst cur
    rw [foldRange]
    simp only [Nat.one_ne_zero,lt_self_iff_false,↓reduceDIte]
termination_by limit r-cur
decreasing_by omega

theorem accumulated_prefix [Inhabited Word] (m : Arithmetic Word r) (g : Graph)
    (ba bb pid : Nat) (s : State Word) :
    Execution.accumulated m g ba bb pid s=accumulatorAt m g ba bb pid s (limit r) := by
  simpa only [Execution.accumulated,Execution.sourceRange_eq,accumulatorAt,limit] using
    range_prefix m g ba bb pid s 0 (Nat.zero_le _)

/-- Every field concerns one reached primitive call. Tile arguments include
masked zeros; accumulators are the actual preceding primitive results. No field
assumes a kernel result, execution success, address, load, store, or frame law. -/
structure Calls [Inhabited Word] (actual expected : Arithmetic Word r) (g : Graph)
    (ba bb pid : Nat) (s : State Word) : Prop where
  zero : actual.zero=expected.zero
  dot : g=.source → ∀ k,k<limit r →
    actual.dot (Execution.aLoad expected ba bb pid k s) (Execution.bLoad expected ba bb pid k s)=
    expected.dot (Execution.aLoad expected ba bb pid k s) (Execution.bLoad expected ba bb pid k s)
  add : g=.source → ∀ k,k<limit r → ∀ q,
    actual.add (accumulatorAt expected g ba bb pid s k q)
      (expected.dot (Execution.aLoad expected ba bb pid k s) (Execution.bLoad expected ba bb pid k s) q)=
    expected.add (accumulatorAt expected g ba bb pid s k q)
      (expected.dot (Execution.aLoad expected ba bb pid k s) (Execution.bLoad expected ba bb pid k s) q)
  dotAcc : g=.accumulator → ∀ k,k<limit r →
    actual.dotAcc (Execution.aLoad expected ba bb pid k s) (Execution.bLoad expected ba bb pid k s)
      (accumulatorAt expected g ba bb pid s k)=
    expected.dotAcc (Execution.aLoad expected ba bb pid k s) (Execution.bLoad expected ba bb pid k s)
      (accumulatorAt expected g ba bb pid s k)
  narrow : ∀ q,actual.narrow (accumulatorAt expected g ba bb pid s (limit r) q)=
    expected.narrow (accumulatorAt expected g ba bb pid s (limit r) q)
  widen : ∀ q,actual.widen (expected.narrow (accumulatorAt expected g ba bb pid s (limit r) q))=
    expected.widen (expected.narrow (accumulatorAt expected g ba bb pid s (limit r) q))

private theorem loads_equal [Inhabited Word] (actual expected : Arithmetic Word r)
    (ba bb pid k : Nat) (s : State Word) (hz : actual.zero=expected.zero) :
    Execution.aLoad actual ba bb pid k s=Execution.aLoad expected ba bb pid k s ∧
    Execution.bLoad actual ba bb pid k s=Execution.bLoad expected ba bb pid k s := by
  constructor <;> funext q <;> simp only [Execution.aLoad,Execution.bLoad,hz]

theorem prefix_equal [Inhabited Word] (actual expected : Arithmetic Word r) (g : Graph)
    (ba bb pid : Nat) (s : State Word) (law : Calls actual expected g ba bb pid s)
    (k : Nat) (hk : k≤limit r) :
    accumulatorAt actual g ba bb pid s k=accumulatorAt expected g ba bb pid s k := by
  induction k with
  | zero => simp only [accumulatorAt,law.zero]
  | succ k ih =>
    have hl := loads_equal actual expected ba bb pid k s law.zero
    simp only [accumulatorAt,ih (by omega)]
    cases g with
    | source =>
      simp only [Execution.step,hl.1,hl.2,law.dot rfl k (by omega)]
      funext q
      exact law.add rfl k (by omega) q
    | accumulator =>
      simp only [Execution.step,hl.1,hl.2]
      exact law.dotAcc rfl k (by omega)

theorem program_equal [Inhabited Word] (actual expected : Arithmetic Word r) (g : Graph)
    (ba bb bc pid : Nat) (s : State Word) (law : Calls actual expected g ba bb pid s) :
    Execution.program actual g ba bb bc pid s=Execution.program expected g ba bb bc pid s := by
  unfold Execution.program
  rw [accumulated_prefix,accumulated_prefix,prefix_equal actual expected g ba bb pid s law (limit r) (Nat.le_refl _)]
  dsimp only
  have casts : (fun q => actual.widen (actual.narrow (accumulatorAt expected g ba bb pid s (limit r) q)))=
      (fun q => expected.widen (expected.narrow (accumulatorAt expected g ba bb pid s (limit r) q))) := by
    funext q
    rw [law.narrow q,law.widen q]
  simp only [show ∀ q,actual.widen (actual.narrow (accumulatorAt expected g ba bb pid s (limit r) q))=
      expected.widen (expected.narrow (accumulatorAt expected g ba bb pid s (limit r) q)) from fun q => congrFun casts q]

/-- Calls are instantiated only in the actual ordered program instances and
with the evolving state left by preceding expected instances. -/
def LaunchCalls [Inhabited Word] (actual expected : Arithmetic Word r) (g : Graph)
    (ba bb bc : Nat) : List Nat → State Word → Prop
  | [],_ => True
  | pid::ids,s => Calls actual expected g ba bb pid s ∧
    LaunchCalls actual expected g ba bb bc ids (Execution.program expected g ba bb bc pid s)

theorem programs_equal [Inhabited Word] (actual expected : Arithmetic Word r) (g : Graph)
    (ba bb bc : Nat) (ids : List Nat) (s : State Word)
    (law : LaunchCalls actual expected g ba bb bc ids s) :
    Execution.runPrograms actual g ba bb bc ids s=Execution.runPrograms expected g ba bb bc ids s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid ids ih =>
    simp only [LaunchCalls] at law
    simp only [Execution.runPrograms,program_equal actual expected g ba bb bc pid s law.1]
    exact ih _ law.2

theorem host_equal [Inhabited Word] (actual expected : Arithmetic Word r) (g : Graph)
    (ba bb bc : Nat) (s : State Word)
    (law : LaunchCalls actual expected g ba bb bc (List.range (programs r)) s) :
    Execution.host actual g ba bb bc s=Execution.host expected g ba bb bc s := by
  unfold Execution.host
  rw [programs_equal actual expected g ba bb bc _ s law]

/-- The same local calls, now at the actual allocation results of each reached
successful wrapper launch. The antecedents select reached API states; the
conclusion contains only the six primitive bit equalities from Calls. -/
def WrapperCalls [Inhabited Word] {Device Error : Type}
    (actual expected : (r : Request) → Arithmetic Word r) (g : Graph)
    (api : Wrapper.Calls Word Device Error) (r : Request) (before : Wrapper.Heap Word) : Prop :=
  ∀ d (p : Wrapper.Prepared Word r before), Wrapper.PreparedBy api r before d p →
    ∀ support : Index64.Launchable p.request, Wrapper.launch api d p=.ok ⟨support⟩ →
      LaunchCalls (actual p.request) (expected p.request) g p.a.tensor.base p.b.tensor.base
        p.c.tensor.base (List.range (programs p.request)) p.c.after.state

/-- Local primitive correspondence composes through the complete wrapper,
including exact ordinary error outcomes and the same returned object/heap. -/
theorem wrapper_equal [Inhabited Word] {Device Error : Type}
    (actual expected : (r : Request) → Arithmetic Word r) (g : Graph)
    (api : Wrapper.Calls Word Device Error) (r : Request) (before : Wrapper.Heap Word)
    (law : WrapperCalls actual expected g api r before) :
    Wrapper.run actual g api r before=Wrapper.run expected g api r before := by
  rcases Wrapper.prepare_cases api r before with ⟨e,he,_⟩ | ⟨d,p,hp,ht⟩
  · simp only [Wrapper.run,he]
  · cases hl : Wrapper.launch api d p with
    | error e => simp only [Wrapper.run,hp,hl]
    | ok support =>
      have hh := host_equal (actual p.request) (expected p.request) g
        p.a.tensor.base p.b.tensor.base p.c.tensor.base p.c.after.state
        (law d p ht support.down hl)
      simp only [Wrapper.run,hp,hl,Wrapper.result,hh]

#axiomsClean wrapper_equal
#axiomsClean prefix_equal
#axiomsClean program_equal
#axiomsClean host_equal
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IvDependentMatmul.Conformance
