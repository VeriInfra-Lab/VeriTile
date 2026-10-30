import bench.tritonbench_g.spinning_lock_reduction.Proofs.Termination
import bench.tritonbench_g.spinning_lock_reduction.Proofs.HostFixtures

noncomputable section
open Classical
namespace VeriTile.Bench.TritonBenchG.SpinningLock
open Kernels

/-- Independent mathematical rectangle in the caller's strided C memory. -/
def mathematicalFootprint (h : HostArgs) (a : Int) : Prop :=
  ∃ row < min h.bm h.m, ∃ col < min (h.numSms * h.bn) h.n,
    (row * h.strideM + col * h.strideN : Nat) = a

theorem source_footprint_eq_mathematical (h : HostArgs) (buf : HostBuffers h) (a : Int) :
    h.outputFootprint a ↔ mathematicalFootprint h a := by
  constructor
  · rintro ⟨p, hp, l, hl, hm, ha⟩
    have hl' : l < h.bm * h.bn := hl
    have hrow : l / h.bn < h.bm := (Nat.div_lt_iff_lt_mul buf.tileNPositive).mpr hl'
    have hrem : l % h.bn < h.bn := Nat.mod_lt _ buf.tileNPositive
    have hcol : p * h.bn + l % h.bn < h.numSms * h.bn := by
      have hh := Nat.mul_le_mul_right h.bn (show p + 1 ≤ h.numSms by omega)
      simp only [Nat.add_mul, Nat.one_mul] at hh
      omega
    obtain ⟨hrm, hcn⟩ := (buf.maskExact p hp l hl).mp hm
    refine ⟨l / h.bn, Nat.lt_min.mpr ⟨hrow, hrm⟩,
      p * h.bn + l % h.bn, Nat.lt_min.mpr ⟨hcol, hcn⟩, ?_⟩
    rw [← buf.outputOffset p hp l hl]
    exact ha
  · rintro ⟨row, hr, col, hc, ha⟩
    obtain ⟨hrb, hrm⟩ := Nat.lt_min.mp hr
    obtain ⟨hcs, hcn⟩ := Nat.lt_min.mp hc
    let p := col / h.bn
    let l := row * h.bn + col % h.bn
    have hp : p < h.numSms := (Nat.div_lt_iff_lt_mul buf.tileNPositive).mpr hcs
    have hrem : col % h.bn < h.bn := Nat.mod_lt _ buf.tileNPositive
    have hl : l < h.config.lanes := by
      have hh := Nat.mul_le_mul_right h.bn (show row + 1 ≤ h.bm by omega)
      simp only [Nat.add_mul, Nat.one_mul] at hh
      change row * h.bn + col % h.bn < h.bm * h.bn
      omega
    have hrow : l / h.bn = row := by
      dsimp [l]
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ buf.tileNPositive,
        Nat.div_eq_of_lt hrem, Nat.zero_add]
    have hcol : p * h.bn + l % h.bn = col := by
      dsimp [p, l]
      simp only [Nat.add_mod, Nat.mul_mod_left, Nat.zero_add, Nat.mod_mod]
      simpa [Nat.mul_comm] using Nat.div_add_mod col h.bn
    refine ⟨p, hp, l, hl, (buf.maskExact p hp l hl).mpr ?_, ?_⟩
    · simpa [hrow, hcol] using And.intro hrm hcn
    · rw [buf.outputOffset p hp l hl, hrow, hcol]
      exact ha

/-- Complete launch facts retain every prefix fact and derive termination. -/
structure CompleteFacts (b : Backend V) (h : HostArgs) (buf : HostBuffers h)
    (schedule : List Nat) (mem : Memory V) (result : WrapperResult V) : Prop
    extends FreshFacts b h buf schedule mem result where
  done : allDone h.config result.final = true

theorem CompleteFacts.mathematical_output (facts : CompleteFacts b h buf schedule mem result) (a : Int) :
    result.final.mem.output a = if mathematicalFootprint h a then b.zero else mem.output a := by
  rw [fresh_completed_output facts.toFreshFacts facts.done a,
    source_footprint_eq_mathematical h buf a]

theorem ready_complete_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (buf : HostBuffers h) (mem : Memory V) (ready : PublishedReady b h.config mem) :
    ∃ schedule result, CompleteFacts b h buf schedule mem result := by
  obtain ⟨schedule, hd⟩ := source_schedule_completes b h buf mem
  obtain ⟨result, facts⟩ := ready_source_backend b hadd h buf schedule mem ready
  have he := Option.some.inj (facts.execution.symm.trans (launch_execution b h schedule mem))
  refine ⟨schedule, result, ⟨facts, ?_⟩⟩
  simpa only [he] using hd

theorem fresh_complete_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (buf : HostBuffers h) (mem : Memory V) (fresh : ∀ a, mem.locks a = 0) :
    ∃ schedule result, CompleteFacts b h buf schedule mem result :=
  ready_complete_backend b hadd h buf mem (fresh_publishedReady b h.config mem fresh)

theorem relaunch_complete_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (previous : CompleteFacts b oldHost oldBuf oldSchedule oldMem oldResult)
    (h : HostArgs) (buf : HostBuffers h) (hbm : h.bm = oldHost.bm) (hbn : h.bn = oldHost.bn) :
    ∃ schedule result, CompleteFacts b h buf schedule oldResult.final.mem result := by
  apply ready_complete_backend b hadd h buf oldResult.final.mem
  intro q hq l hl
  have hold := previous.ready q hq l
  simpa only [Config.pAddr, Config.lanes, HostArgs.config, hbm, hbn] using
    hold (by simpa only [Config.lanes, HostArgs.config, hbm, hbn] using hl)

/-- Exact four wrapper calls; every transition consumes the previous in-place
memory. The caller's three allocations remain the original sizes throughout. -/
structure OriginalFourFacts (b : Backend V) (mem : Memory V)
    (s₁ s₂ s₃ s₄ : List Nat) (r₁ r₂ r₃ r₄ : WrapperResult V) : Prop where
  first : CompleteFacts b (fixtureArgs 304 3)
    (fixtureBuffers 304 3 (by decide) (by decide) (by decide) (by decide)) s₁ mem r₁
  second : CompleteFacts b (fixtureArgs 304 2)
    (fixtureBuffers 304 2 (by decide) (by decide) (by decide) (by decide)) s₂ r₁.final.mem r₂
  third : CompleteFacts b (fixtureArgs 2 2)
    (fixtureBuffers 2 2 (by decide) (by decide) (by decide) (by decide)) s₃ r₂.final.mem r₃
  fourth : CompleteFacts b (fixtureArgs 5 2)
    (fixtureBuffers 5 2 (by decide) (by decide) (by decide) (by decide)) s₄ r₃.final.mem r₄

theorem original_four_complete_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (mem : Memory V) (fresh : ∀ a, mem.locks a = 0) :
    ∃ s₁ s₂ s₃ s₄ r₁ r₂ r₃ r₄, OriginalFourFacts b mem s₁ s₂ s₃ s₄ r₁ r₂ r₃ r₄ := by
  obtain ⟨s₁, r₁, f₁⟩ := fresh_complete_backend b hadd (fixtureArgs 304 3)
    (fixtureBuffers 304 3 (by decide) (by decide) (by decide) (by decide)) mem fresh
  obtain ⟨s₂, r₂, f₂⟩ := relaunch_complete_backend b hadd f₁ (fixtureArgs 304 2)
    (fixtureBuffers 304 2 (by decide) (by decide) (by decide) (by decide)) rfl rfl
  obtain ⟨s₃, r₃, f₃⟩ := relaunch_complete_backend b hadd f₂ (fixtureArgs 2 2)
    (fixtureBuffers 2 2 (by decide) (by decide) (by decide) (by decide)) rfl rfl
  obtain ⟨s₄, r₄, f₄⟩ := relaunch_complete_backend b hadd f₃ (fixtureArgs 5 2)
    (fixtureBuffers 5 2 (by decide) (by decide) (by decide) (by decide)) rfl rfl
  exact ⟨s₁, s₂, s₃, s₄, r₁, r₂, r₃, r₄, ⟨f₁, f₂, f₃, f₄⟩⟩

theorem fixture_footprint_subset (sms k : Nat) (hs : sms ≤ 304) (a : Int) :
    mathematicalFootprint (fixtureArgs sms k) a → mathematicalFootprint (fixtureArgs 304 3) a := by
  rintro ⟨row, hr, col, hc, ha⟩
  refine ⟨row, hr, col, ?_, ha⟩
  dsimp [fixtureArgs] at hc ⊢
  omega

theorem CompleteFacts.preserve_target {V : Type} {b : Backend V}
    {h : HostArgs} {buf : HostBuffers h} {schedule : List Nat}
    {previous : Memory V} {result : WrapperResult V}
    (facts : CompleteFacts b h buf schedule previous result)
    (target : Int → Prop) (initialC : Int → V)
    (hprevious : ∀ a, previous.output a = if target a then b.zero else initialC a)
    (hsub : ∀ a, mathematicalFootprint h a → target a) :
    ∀ a, result.final.mem.output a = if target a then b.zero else initialC a := by
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
      r₁.final.mem.output a = value ∧ r₂.final.mem.output a = value ∧
      r₃.final.mem.output a = value ∧ r₄.final.mem.output a = value := by
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
    ∀ a, r₁.final.mem.output a = b.zero ∧ r₂.final.mem.output a = b.zero ∧
      r₃.final.mem.output a = b.zero ∧ r₄.final.mem.output a = b.zero := by
  intro a
  simpa only [hzero, ite_self] using facts.snapshots a

#axiomsClean source_footprint_eq_mathematical
#axiomsClean fresh_complete_backend
#axiomsClean original_four_complete_backend
#axiomsClean OriginalFourFacts.snapshots
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SpinningLock
