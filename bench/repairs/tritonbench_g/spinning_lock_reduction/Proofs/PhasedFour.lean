import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.PhasedTermination

noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.Phased
open VeriTile.Bench.TritonBenchG.SpinningLock Wide

theorem ready_complete_plain (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (buf : Buffers h) (mem : Memory V) (ready : PublishedReady b h mem) :
    ∃ plan result, Synchronized b h plan mem ∧
      Wide.CompleteFacts b h buf (flatten h plan) mem result := by
  obtain ⟨plan,sync⟩ := source_phases_complete b h buf mem
  refine ⟨plan, Wide.launch b h (flatten h plan) mem, sync,
    ⟨ready_prefix_backend b hadd h buf _ mem ready, ?_⟩⟩
  rw [launch_eq_run b h buf]
  exact sync.all_done

theorem relaunch_complete_plain (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (previous : Wide.CompleteFacts b oldHost oldBuf oldSchedule oldMem oldResult)
    (h : HostArgs) (buf : Buffers h) (hbm : h.bm = oldHost.bm) (hbn : h.bn = oldHost.bn) :
    ∃ plan result, Synchronized b h plan oldResult.final.values.mem ∧
      Wide.CompleteFacts b h buf (flatten h plan) oldResult.final.values.mem result := by
  apply ready_complete_plain b hadd h buf oldResult.final.values.mem
  intro q hq l hl
  have hold := previous.invariant.ready q hq l
  simpa only [pAddress, lanes, hbm, hbn] using hold (by simpa only [lanes, hbm, hbn] using hl)

structure FourFacts (b : Backend V) (mem : Memory V)
    (p₁ p₂ p₃ p₄ : Plan) (r₁ r₂ r₃ r₄ : WrapperResult V) : Prop where
  values : OriginalFourFacts b mem (flatten (fixtureArgs 304 3) p₁)
    (flatten (fixtureArgs 304 2) p₂) (flatten (fixtureArgs 2 2) p₃)
    (flatten (fixtureArgs 5 2) p₄) r₁ r₂ r₃ r₄
  first : Synchronized b (fixtureArgs 304 3) p₁ mem
  second : Synchronized b (fixtureArgs 304 2) p₂ r₁.final.values.mem
  third : Synchronized b (fixtureArgs 2 2) p₃ r₂.final.values.mem
  fourth : Synchronized b (fixtureArgs 5 2) p₄ r₃.final.values.mem

theorem four_complete (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (mem : Memory V) (fresh : ∀ a, mem.locks a = 0) :
    ∃ p₁ p₂ p₃ p₄ r₁ r₂ r₃ r₄, FourFacts b mem p₁ p₂ p₃ p₄ r₁ r₂ r₃ r₄ := by
  obtain ⟨p₁,r₁,s₁,f₁⟩ := ready_complete_plain b hadd (fixtureArgs 304 3)
    (fixtureWideBuffers 304 3 (by decide) (by decide) (by decide) (by decide)) mem
    (fresh_ready b _ mem fresh)
  obtain ⟨p₂,r₂,s₂,f₂⟩ := relaunch_complete_plain b hadd f₁ (fixtureArgs 304 2)
    (fixtureWideBuffers 304 2 (by decide) (by decide) (by decide) (by decide)) rfl rfl
  obtain ⟨p₃,r₃,s₃,f₃⟩ := relaunch_complete_plain b hadd f₂ (fixtureArgs 2 2)
    (fixtureWideBuffers 2 2 (by decide) (by decide) (by decide) (by decide)) rfl rfl
  obtain ⟨p₄,r₄,s₄,f₄⟩ := relaunch_complete_plain b hadd f₃ (fixtureArgs 5 2)
    (fixtureWideBuffers 5 2 (by decide) (by decide) (by decide) (by decide)) rfl rfl
  exact ⟨p₁,p₂,p₃,p₄,r₁,r₂,r₃,r₄,⟨⟨f₁,f₂,f₃,f₄⟩,s₁,s₂,s₃,s₄⟩⟩

theorem FourFacts.physical_calls (encoding : SignEncoding V)
    (plain : ∀ x, encoding.flip false x = x)
    (facts : FourFacts b mem p₁ p₂ p₃ p₄ r₁ r₂ r₃ r₄) :
    r₁ = observe b encoding {} (fixtureArgs 304 3) p₁ mem ∧
    r₂ = observe b encoding {} (fixtureArgs 304 2) p₂ r₁.final.values.mem ∧
    r₃ = observe b encoding {} (fixtureArgs 2 2) p₃ r₂.final.values.mem ∧
    r₄ = observe b encoding {} (fixtureArgs 5 2) p₄ r₃.final.values.mem :=
  facts.values.physical_calls encoding plain

#axiomsClean four_complete
#axiomsClean FourFacts.physical_calls
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Phased
