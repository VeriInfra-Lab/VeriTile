import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.WideViews
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.WideFour

noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.Wide
open VeriTile.Bench.TritonBenchG.SpinningLock VeriTile.Triton

theorem flip_injective (encoding : SignEncoding V) (flag : Bool) : Function.Injective (encoding.flip flag) := by
  intro x y he
  have hh := congrArg (encoding.flip flag) he
  simpa only [encoding.twice] using hh

theorem lockWord_injective (flag : Bool) : Function.Injective (lockWord flag) := by
  intro x y he
  have hh := congrArg (lockWord flag) he
  simpa only [lockWord_twice] using hh

/-- Full physical backing frame follows from the logical frame and explicit
involutions. The identity of every supplied allocation and every untouched cell
is retained, even when a negative view is passed for an output/scratch role. -/
theorem physical_frame (encoding : SignEncoding V) (flags : ViewFlags)
    (raw : Memory V) (result : WrapperResult V)
    (facts : PrefixFacts b h buf schedule (interpretMemory encoding flags raw)
      (interpretResult encoding flags result)) (a : Int) :
    ((∀ p l, Event.publish p l a ∉ result.events) → result.final.values.mem.partials a = raw.partials a) ∧
    ((∀ p l, Event.output p l a ∉ result.events) → result.final.values.mem.output a = raw.output a) ∧
    (Event.release a ∉ result.events → result.final.values.mem.locks a = raw.locks a) := by
  constructor
  · intro he
    apply flip_injective encoding flags.partialNeg
    exact facts.partialFrame a he
  constructor
  · intro he
    apply flip_injective encoding flags.output
    exact facts.outputFrame a he
  · intro he
    apply lockWord_injective flags.locks
    exact facts.lockFrame a he

theorem physical_prefix_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (encoding : SignEncoding V) (flags : ViewFlags) (h : HostArgs) (buf : Buffers h)
    (schedule : List Nat) (raw : Memory V)
    (fresh : ∀ a, (interpretMemory encoding flags raw).locks a = 0) :
    PrefixFacts b h buf schedule (interpretMemory encoding flags raw)
      (interpretResult encoding flags (physicalLaunch b encoding flags h schedule raw)) := by
  simp only [physicalLaunch, interpretResult_twice]
  exact ready_prefix_backend b hadd h buf schedule _ (fresh_ready b h _ fresh)

theorem interpretMemory_plain (encoding : SignEncoding V) (plain : ∀ x, encoding.flip false x = x)
    (mem : Memory V) : interpretMemory encoding {} mem = mem := by
  cases mem
  simp [interpretMemory, lockWord, plain]

theorem interpretResult_plain (encoding : SignEncoding V) (plain : ∀ x, encoding.flip false x = x)
    (result : WrapperResult V) : interpretResult encoding {} result = result := by
  cases result with
  | mk returned events final =>
    cases final with
    | mk values pc =>
      cases values
      simp [interpretResult, interpretState, interpretValues, interpretMemory_plain encoding plain]

theorem physicalLaunch_plain (b : Backend V) (encoding : SignEncoding V)
    (plain : ∀ x, encoding.flip false x = x) (h : HostArgs) (schedule : List Nat) (mem : Memory V) :
    physicalLaunch b encoding {} h schedule mem = launch b h schedule mem := by
  simp [physicalLaunch, interpretMemory_plain encoding plain, interpretResult_plain encoding plain]

theorem OriginalFourFacts.physical_calls (encoding : SignEncoding V)
    (plain : ∀ x, encoding.flip false x = x)
    (facts : OriginalFourFacts b mem s₁ s₂ s₃ s₄ r₁ r₂ r₃ r₄) :
    r₁ = physicalLaunch b encoding {} (fixtureArgs 304 3) s₁ mem ∧
    r₂ = physicalLaunch b encoding {} (fixtureArgs 304 2) s₂ r₁.final.values.mem ∧
    r₃ = physicalLaunch b encoding {} (fixtureArgs 2 2) s₃ r₂.final.values.mem ∧
    r₄ = physicalLaunch b encoding {} (fixtureArgs 5 2) s₄ r₃.final.values.mem := by
  simp only [physicalLaunch_plain b encoding plain]
  exact ⟨facts.first.execution, facts.second.execution, facts.third.execution, facts.fourth.execution⟩

#axiomsClean physical_frame
#axiomsClean physical_prefix_backend
#axiomsClean OriginalFourFacts.physical_calls
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Wide
