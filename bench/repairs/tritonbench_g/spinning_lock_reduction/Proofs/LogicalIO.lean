import bench.repairs.tritonbench_g.spinning_lock_reduction.IO
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.IOImplements
namespace VeriTile.Bench.Repairs.SpinningLock.Logical
open VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock
open Classical
noncomputable section
/-- The exact complete concurrent judgment is retained for any logical stage function. -/
theorem judgment_iff (h : HostArgs) (count : Nat) (stage : Phased.Stage)
    (f : Formula.Values ℝ h count → Formula.Values ℝ h count) :
    (io h count stage).ImplementsOn (Formula.Domain 0 h stage) f ↔
    (IO.io ℝ h count stage).ImplementsOn (ConcurrentSource.backend realBackend)
      (IO.domain 0 h count stage) (fun xs => IO.outputs h count stage (f (IO.inputs h count stage xs))) := Iff.rfl
private def readerStage (h : HostArgs) (count : Nat) (stage : Phased.Stage)
    (input : Formula.Values ℝ h count) : Formula.Values ℝ h count := {
    partials := fun i =>
      if stage = .producers ∧ (i.val / (h.bm*h.bn)) % h.groupK ≠ 0 then 0 else input.partials i
    output := fun i =>
      if ∃ p < h.numSms,
          (match stage with | .producers => p % h.groupK ≠ 0 | .consumers => p % h.groupK = 0) ∧
          ∃ row < h.bm, ∃ col < h.bn, row < h.m ∧ p*h.bn+col < h.n ∧
            row*h.strideM+(p*h.bn+col)*h.strideN = i.val
      then 0 else input.output i
    locks := fun i => if stage = .producers ∧ i.val % h.groupK ≠ 0 then 1 else input.locks i }

/-- This is the concrete old reader formula connection, in addition to the
coordinate-only generic iff. All producer/consumer cases are discharged here. -/
theorem reader_formula_eq (h : HostArgs) (count : Nat) (stage : Phased.Stage) :
    readerStage h count stage = Formula.stage 0 h stage := by
  cases stage <;> funext input <;>
    simp only [readerStage,Formula.stage,Formula.writes,Formula.role] <;> congr!

theorem reader_judgment_iff (h : HostArgs) (count : Nat) (stage : Phased.Stage) :
    (io h count stage).ImplementsOn (Formula.Domain 0 h stage) (readerStage h count stage) ↔
    (IO.io ℝ h count stage).ImplementsOn (ConcurrentSource.backend realBackend)
      (IO.domain 0 h count stage) (IO.mathematical 0 h count stage) := by
  apply (judgment_iff h count stage (readerStage h count stage)).trans
  simp only [reader_formula_eq]
  rfl

theorem reader_implements (h : HostArgs) (buf : Wide.Buffers h) (stage : Phased.Stage) :
    (io h buf.cElements stage).ImplementsOn (Formula.Domain 0 h stage)
      (readerStage h buf.cElements stage) :=
  (reader_judgment_iff h buf.cElements stage).2
    (IOProof.implements realBackend (by simp [realBackend]) h buf stage)

end
end VeriTile.Bench.Repairs.SpinningLock.Logical
