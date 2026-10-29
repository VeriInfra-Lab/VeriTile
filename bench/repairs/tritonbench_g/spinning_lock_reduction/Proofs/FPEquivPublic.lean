import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.Model
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.Publication
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.PhasedFPAdapter

/-! Independent exact FP32 mathematical target on the strided rectangle.
Pending/failing executions have finite-prefix semantics; successful submission
returns None asynchronously. Only synchronized completion promises the complete
rectangle. Same-stream producer completion makes every consumer poll ready.
No numerical assumptions are deferred. -/
noncomputable section
open Classical
namespace VeriTile.Bench.Repairs.SpinningLock.FPEquiv
open VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock

def positiveZero : FP.Value .fp32 := FP.Value.zero .fp32

def completeTarget (h : HostArgs) (logicalInput : Memory (FP.Value .fp32)) (a : Int) : (FP.Value .fp32) :=
  if mathematicalFootprint h a then positiveZero else logicalInput.output a

specification repaired_fp32_equivalence (h : HostArgs) (buf : Wide.Buffers h)
    (flags : Wide.ViewFlags) (host : Phased.HostResult) (plan : Phased.Plan) (raw : Memory (FP.Value .fp32))
    (allowed : Phased.AllowedExecution fp32 h host plan (Wide.interpretMemory Wide.fpEncoding flags raw))
    (fresh : ∀ a, (Wide.interpretMemory Wide.fpEncoding flags raw).locks a = 0) :
    let result := Kernels.candidateObservation fp32 Wide.fpEncoding flags h plan raw
    let logical := Wide.interpretResult Wide.fpEncoding flags result
    Wide.PrefixFacts fp32 h buf (Phased.flatten h plan) (Wide.interpretMemory Wide.fpEncoding flags raw) logical ∧
      (∀ a, logical.final.values.mem.output a =
        if Visited result.events a then positiveZero else (Wide.interpretMemory Wide.fpEncoding flags raw).output a) ∧
      (∀ a, (∀ p l, Event.output p l a ∉ result.events) → result.final.values.mem.output a = raw.output a) := by
  have facts := Phased.allowed_prefix_backend fp32 fp32_positive_zero_add
    Wide.fpEncoding flags h buf host plan raw allowed (Wide.fresh_ready fp32 h _ fresh)
  exact ⟨facts, facts.output, fun a => (Wide.physical_frame Wide.fpEncoding flags raw _ facts a).2.1⟩

specification repaired_fp32_synchronized (h : HostArgs) (buf : Wide.Buffers h)
    (flags : Wide.ViewFlags) (plan : Phased.Plan) (raw : Memory (FP.Value .fp32))
    (fresh : ∀ a, (Wide.interpretMemory Wide.fpEncoding flags raw).locks a = 0)
    (sync : Phased.Synchronized fp32 h plan (Wide.interpretMemory Wide.fpEncoding flags raw)) :
    let result := Kernels.candidateObservation fp32 Wide.fpEncoding flags h plan raw
    Wide.CompleteFacts fp32 h buf (Phased.flatten h plan) (Wide.interpretMemory Wide.fpEncoding flags raw)
      (Wide.interpretResult Wide.fpEncoding flags result) ∧
      ∀ a, (Wide.interpretResult Wide.fpEncoding flags result).final.values.mem.output a =
        completeTarget h (Wide.interpretMemory Wide.fpEncoding flags raw) a := by
  have facts := Phased.synchronized_complete_backend fp32 fp32_positive_zero_add Wide.fpEncoding
    flags h buf plan raw (Wide.fresh_ready fp32 h _ fresh) sync
  exact ⟨facts, facts.mathematical_output⟩

specification repaired_fp32_complete (h : HostArgs) (buf : Wide.Buffers h)
    (flags : Wide.ViewFlags) (raw : Memory (FP.Value .fp32))
    (fresh : ∀ a, (Wide.interpretMemory Wide.fpEncoding flags raw).locks a = 0) :
    ∃ plan, Phased.Synchronized fp32 h plan (Wide.interpretMemory Wide.fpEncoding flags raw) ∧
      ∀ a, (Wide.interpretResult Wide.fpEncoding flags
        (Kernels.candidateObservation fp32 Wide.fpEncoding flags h plan raw)).final.values.mem.output a =
          completeTarget h (Wide.interpretMemory Wide.fpEncoding flags raw) a := by
  obtain ⟨plan,sync,facts⟩ := Phased.complete_backend fp32 fp32_positive_zero_add Wide.fpEncoding
    flags h buf raw (Wide.fresh_ready fp32 h _ fresh)
  exact ⟨plan,sync,facts.mathematical_output⟩

specification repaired_four_fp32_complete (mem : Memory (FP.Value .fp32)) (fresh : ∀ a, mem.locks a = 0) :
    ∃ p₁ p₂ p₃ p₄ r₁ r₂ r₃ r₄,
      Phased.FourFacts fp32 mem p₁ p₂ p₃ p₄ r₁ r₂ r₃ r₄ ∧
      (r₁ = Kernels.candidateObservation fp32 Wide.fpEncoding {} (fixtureArgs 304 3) p₁ mem ∧
       r₂ = Kernels.candidateObservation fp32 Wide.fpEncoding {} (fixtureArgs 304 2) p₂ r₁.final.values.mem ∧
       r₃ = Kernels.candidateObservation fp32 Wide.fpEncoding {} (fixtureArgs 2 2) p₃ r₂.final.values.mem ∧
       r₄ = Kernels.candidateObservation fp32 Wide.fpEncoding {} (fixtureArgs 5 2) p₄ r₃.final.values.mem) ∧
      ∀ a, r₁.final.values.mem.output a = completeTarget (fixtureArgs 304 3) mem a ∧
        r₂.final.values.mem.output a = completeTarget (fixtureArgs 304 3) mem a ∧
        r₃.final.values.mem.output a = completeTarget (fixtureArgs 304 3) mem a ∧
        r₄.final.values.mem.output a = completeTarget (fixtureArgs 304 3) mem a := by
  obtain ⟨p₁,p₂,p₃,p₄,r₁,r₂,r₃,r₄,facts⟩ := Phased.four_complete fp32 fp32_positive_zero_add mem fresh
  exact ⟨p₁,p₂,p₃,p₄,r₁,r₂,r₃,r₄,facts,
    facts.physical_calls Wide.fpEncoding (fun _ => rfl),facts.values.snapshots⟩

specification repaired_full_fp32_equivalence (h : HostArgs) (buf : Wide.Buffers h) (flags : Wide.ViewFlags) :
    VeriTile.Spec.FloatingPoint [] (Phased.FPAdapter.candidate h buf flags) (Phased.FPAdapter.reference h buf flags) :=
  Phased.FPAdapter.full_fp32_equivalence h buf flags

#print_fp_assumptions repaired_full_fp32_equivalence

#axiomsClean repaired_fp32_equivalence
#axiomsClean repaired_fp32_synchronized
#axiomsClean repaired_fp32_complete
#axiomsClean repaired_four_fp32_complete
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.FPEquiv
