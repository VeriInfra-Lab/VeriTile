import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.Model
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.Publication

/-! Independent Real mathematical target on the strided rectangle.
Pending/failing executions have finite-prefix semantics; successful submission
returns None asynchronously. Only synchronized completion promises the complete
rectangle. Same-stream producer completion makes every consumer poll ready.
No numerical assumptions are deferred. -/
noncomputable section
open Classical
namespace VeriTile.Bench.Repairs.SpinningLock.Correct
open VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock

def zeroTarget (_row _col : Nat) : ℝ := 0

def completeTarget (h : HostArgs) (logicalInput : Memory ℝ) (a : Int) : ℝ :=
  if mathematicalFootprint h a then 0 else logicalInput.output a

specification repaired_real_correctness (h : HostArgs) (buf : Wide.Buffers h)
    (flags : Wide.ViewFlags) (host : Phased.HostResult) (plan : Phased.Plan) (raw : Memory ℝ)
    (allowed : Phased.AllowedExecution realBackend h host plan (Wide.interpretMemory Wide.realEncoding flags raw))
    (fresh : ∀ a, (Wide.interpretMemory Wide.realEncoding flags raw).locks a = 0) :
    let result := Kernels.candidateObservation realBackend Wide.realEncoding flags h plan raw
    let logical := Wide.interpretResult Wide.realEncoding flags result
    Wide.PrefixFacts realBackend h buf (Phased.flatten h plan) (Wide.interpretMemory Wide.realEncoding flags raw) logical ∧
      (∀ a, logical.final.values.mem.output a =
        if Visited result.events a then 0 else (Wide.interpretMemory Wide.realEncoding flags raw).output a) ∧
      (∀ a, (∀ p l, Event.output p l a ∉ result.events) → result.final.values.mem.output a = raw.output a) := by
  have facts := Phased.allowed_prefix_backend realBackend (by simp [realBackend])
    Wide.realEncoding flags h buf host plan raw allowed (Wide.fresh_ready realBackend h _ fresh)
  exact ⟨facts, facts.output, fun a => (Wide.physical_frame Wide.realEncoding flags raw _ facts a).2.1⟩

specification repaired_real_synchronized (h : HostArgs) (buf : Wide.Buffers h)
    (flags : Wide.ViewFlags) (plan : Phased.Plan) (raw : Memory ℝ)
    (fresh : ∀ a, (Wide.interpretMemory Wide.realEncoding flags raw).locks a = 0)
    (sync : Phased.Synchronized realBackend h plan (Wide.interpretMemory Wide.realEncoding flags raw)) :
    let result := Kernels.candidateObservation realBackend Wide.realEncoding flags h plan raw
    Wide.CompleteFacts realBackend h buf (Phased.flatten h plan) (Wide.interpretMemory Wide.realEncoding flags raw)
      (Wide.interpretResult Wide.realEncoding flags result) ∧
      ∀ a, (Wide.interpretResult Wide.realEncoding flags result).final.values.mem.output a =
        completeTarget h (Wide.interpretMemory Wide.realEncoding flags raw) a := by
  have facts := Phased.synchronized_complete_backend realBackend (by simp [realBackend]) Wide.realEncoding
    flags h buf plan raw (Wide.fresh_ready realBackend h _ fresh) sync
  exact ⟨facts, facts.mathematical_output⟩

specification repaired_real_complete (h : HostArgs) (buf : Wide.Buffers h)
    (flags : Wide.ViewFlags) (raw : Memory ℝ)
    (fresh : ∀ a, (Wide.interpretMemory Wide.realEncoding flags raw).locks a = 0) :
    ∃ plan, Phased.Synchronized realBackend h plan (Wide.interpretMemory Wide.realEncoding flags raw) ∧
      ∀ a, (Wide.interpretResult Wide.realEncoding flags
        (Kernels.candidateObservation realBackend Wide.realEncoding flags h plan raw)).final.values.mem.output a =
          completeTarget h (Wide.interpretMemory Wide.realEncoding flags raw) a := by
  obtain ⟨plan,sync,facts⟩ := Phased.complete_backend realBackend (by simp [realBackend]) Wide.realEncoding
    flags h buf raw (Wide.fresh_ready realBackend h _ fresh)
  exact ⟨plan,sync,facts.mathematical_output⟩

specification repaired_four_real_complete (mem : Memory ℝ) (fresh : ∀ a, mem.locks a = 0) :
    ∃ p₁ p₂ p₃ p₄ r₁ r₂ r₃ r₄,
      Phased.FourFacts realBackend mem p₁ p₂ p₃ p₄ r₁ r₂ r₃ r₄ ∧
      (r₁ = Kernels.candidateObservation realBackend Wide.realEncoding {} (fixtureArgs 304 3) p₁ mem ∧
       r₂ = Kernels.candidateObservation realBackend Wide.realEncoding {} (fixtureArgs 304 2) p₂ r₁.final.values.mem ∧
       r₃ = Kernels.candidateObservation realBackend Wide.realEncoding {} (fixtureArgs 2 2) p₃ r₂.final.values.mem ∧
       r₄ = Kernels.candidateObservation realBackend Wide.realEncoding {} (fixtureArgs 5 2) p₄ r₃.final.values.mem) ∧
      ∀ a, r₁.final.values.mem.output a = completeTarget (fixtureArgs 304 3) mem a ∧
        r₂.final.values.mem.output a = completeTarget (fixtureArgs 304 3) mem a ∧
        r₃.final.values.mem.output a = completeTarget (fixtureArgs 304 3) mem a ∧
        r₄.final.values.mem.output a = completeTarget (fixtureArgs 304 3) mem a := by
  obtain ⟨p₁,p₂,p₃,p₄,r₁,r₂,r₃,r₄,facts⟩ := Phased.four_complete realBackend (by simp [realBackend]) mem fresh
  exact ⟨p₁,p₂,p₃,p₄,r₁,r₂,r₃,r₄,facts,
    facts.physical_calls Wide.realEncoding (fun _ => rfl),facts.values.snapshots⟩

#axiomsClean repaired_real_correctness
#axiomsClean repaired_real_synchronized
#axiomsClean repaired_real_complete
#axiomsClean repaired_four_real_complete
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Correct
