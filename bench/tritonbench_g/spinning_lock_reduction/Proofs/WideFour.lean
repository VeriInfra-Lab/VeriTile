import bench.tritonbench_g.spinning_lock_reduction.Proofs.WideCompletion

noncomputable section
open Classical
namespace VeriTile.Bench.Repairs.SpinningLock.Wide
open VeriTile.Bench.TritonBenchG.SpinningLock

/-- Original four-call tensor allocation metadata. Only ordinary capacity and
separation fields are reused; no old fixed-width offset hypotheses are imported. -/
def fixtureWideBuffers (sms k : Nat) (hs0 : 0 < sms) (hs : sms ≤ 304)
    (hk0 : 0 < k) (hk : k ≤ 3) : Buffers (fixtureArgs sms k) where
  pElements := 304 * 128 * 128
  cElements := 1024 * 1024
  lockElements := 304
  pBase := 0
  cBase := 4 * (304 * 128 * 128)
  lockBase := 4 * (304 * 128 * 128) + 4 * (1024 * 1024)
  smsPositive := hs0
  smsGridLimit := by dsimp [fixtureArgs]; omega
  groupPositive := hk0
  groupScalarLimit := by dsimp [fixtureArgs]; omega
  tileMPositive := by dsimp [fixtureArgs]; decide
  tileNPositive := by dsimp [fixtureArgs]; decide
  tileMArange := by dsimp [fixtureArgs]; decide
  tileNArange := by dsimp [fixtureArgs]; decide
  mMetadata := by dsimp [fixtureArgs]; decide
  nMetadata := by dsimp [fixtureArgs]; decide
  strideMMetadata := by dsimp [fixtureArgs]; decide
  strideNMetadata := by dsimp [fixtureArgs]; decide
  partialCapacity := (fixtureBuffers sms k hs0 hs hk0 hk).partialCapacity
  lockCapacity := hs
  outputCapacity := (fixtureBuffers sms k hs0 hs hk0 hk).outputCapacity
  partialStorageBytes := by decide
  outputStorageBytes := by decide
  lockStorageBytes := by decide
  disjointPC := by decide
  disjointPL := by decide
  disjointCL := by decide

/-- Exact four wrapper calls; every transition consumes the previous in-place
memory. The caller's three allocations remain the original sizes throughout. -/
structure OriginalFourFacts (b : Backend V) (mem : Memory V)
    (s₁ s₂ s₃ s₄ : List Nat) (r₁ r₂ r₃ r₄ : WrapperResult V) : Prop where
  first : CompleteFacts b (fixtureArgs 304 3)
    (fixtureWideBuffers 304 3 (by decide) (by decide) (by decide) (by decide)) s₁ mem r₁
  second : CompleteFacts b (fixtureArgs 304 2)
    (fixtureWideBuffers 304 2 (by decide) (by decide) (by decide) (by decide)) s₂ r₁.final.values.mem r₂
  third : CompleteFacts b (fixtureArgs 2 2)
    (fixtureWideBuffers 2 2 (by decide) (by decide) (by decide) (by decide)) s₃ r₂.final.values.mem r₃
  fourth : CompleteFacts b (fixtureArgs 5 2)
    (fixtureWideBuffers 5 2 (by decide) (by decide) (by decide) (by decide)) s₄ r₃.final.values.mem r₄

theorem original_four_complete_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (mem : Memory V) (fresh : ∀ a, mem.locks a = 0) :
    ∃ s₁ s₂ s₃ s₄ r₁ r₂ r₃ r₄, OriginalFourFacts b mem s₁ s₂ s₃ s₄ r₁ r₂ r₃ r₄ := by
  obtain ⟨s₁, r₁, f₁⟩ := fresh_complete_backend b hadd (fixtureArgs 304 3)
    (fixtureWideBuffers 304 3 (by decide) (by decide) (by decide) (by decide)) mem fresh
  obtain ⟨s₂, r₂, f₂⟩ := relaunch_complete_backend b hadd f₁ (fixtureArgs 304 2)
    (fixtureWideBuffers 304 2 (by decide) (by decide) (by decide) (by decide)) rfl rfl
  obtain ⟨s₃, r₃, f₃⟩ := relaunch_complete_backend b hadd f₂ (fixtureArgs 2 2)
    (fixtureWideBuffers 2 2 (by decide) (by decide) (by decide) (by decide)) rfl rfl
  obtain ⟨s₄, r₄, f₄⟩ := relaunch_complete_backend b hadd f₃ (fixtureArgs 5 2)
    (fixtureWideBuffers 5 2 (by decide) (by decide) (by decide) (by decide)) rfl rfl
  exact ⟨s₁, s₂, s₃, s₄, r₁, r₂, r₃, r₄, ⟨f₁, f₂, f₃, f₄⟩⟩

theorem CompleteFacts.preserve_target {V : Type} {b : Backend V}
    {h : HostArgs} {buf : Buffers h} {schedule : List Nat}
    {previous : Memory V} {result : WrapperResult V}
    (facts : CompleteFacts b h buf schedule previous result)
    (target : Int → Prop) (initialC : Int → V)
    (hprevious : ∀ a, previous.output a = if target a then b.zero else initialC a)
    (hsub : ∀ a, mathematicalFootprint h a → target a) :
    ∀ a, result.final.values.mem.output a = if target a then b.zero else initialC a := by
  intro a
  rw [facts.mathematical_output, hprevious]
  by_cases hc : mathematicalFootprint h a
  · simp [hc, hsub a hc]
  · simp [hc]

/-- The first two calls have the same footprint, and the smaller grids have
subsets. Every C snapshot therefore equals the first call's independent target;
this includes arbitrary initial C, not only the all-zero Python test tensor. -/
theorem OriginalFourFacts.snapshots
    (facts : OriginalFourFacts b mem s₁ s₂ s₃ s₄ r₁ r₂ r₃ r₄) :
    ∀ a, let value := if mathematicalFootprint (fixtureArgs 304 3) a then b.zero else mem.output a
      r₁.final.values.mem.output a = value ∧ r₂.final.values.mem.output a = value ∧
      r₃.final.values.mem.output a = value ∧ r₄.final.values.mem.output a = value := by
  have h₁ := facts.first.mathematical_output
  have h₂ := facts.second.preserve_target _ mem.output h₁
    (fixture_footprint_subset 304 2 (by decide))
  have h₃ := facts.third.preserve_target _ mem.output h₂
    (fixture_footprint_subset 2 2 (by decide))
  have h₄ := facts.fourth.preserve_target _ mem.output h₃
    (fixture_footprint_subset 5 2 (by decide))
  intro a
  exact ⟨h₁ a, h₂ a, h₃ a, h₄ a⟩

theorem OriginalFourFacts.zero_snapshots
    (facts : OriginalFourFacts b mem s₁ s₂ s₃ s₄ r₁ r₂ r₃ r₄)
    (hzero : ∀ a, mem.output a = b.zero) :
    ∀ a, r₁.final.values.mem.output a = b.zero ∧ r₂.final.values.mem.output a = b.zero ∧
      r₃.final.values.mem.output a = b.zero ∧ r₄.final.values.mem.output a = b.zero := by
  intro a
  simpa only [hzero, ite_self] using facts.snapshots a


#axiomsClean original_four_complete_backend
#axiomsClean OriginalFourFacts.snapshots
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Wide
