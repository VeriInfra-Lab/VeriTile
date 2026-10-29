import bench.repairs.tritonbench_g.matrix_transpose.Kernels
import bench.repairs.tritonbench_g.matrix_transpose.Proofs.Indices
import bench.tritonbench_g.matrix_transpose.Proofs.FPSoftware
import bench.tritonbench_g.matrix_transpose.HostCorrect

namespace VeriTile.Bench.Repairs.MatrixTranspose.FP
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixTranspose
noncomputable section

abbrev Word := FPBits.Word

def parameters (m d : Nat) : Host.Input := ⟨"matrix", "out", m, d⟩

/-- Guarded, original single-program source; no value-producing callback. -/
def runWrapper (m d : Nat) (samples : Nat → Word) (initial : State Word) :
    Except InputError (Option (ResultMetadata × State Word)) := do
  checkInputs (.integer m) (.integer d)
  return (Host.run (parameters m d) initial samples).map fun s =>
    (returnedTensor m d, s)

theorem source_is_original (m d : Nat) :
    source "matrix" "out" m d = FPBits.source Bool.false (Host.parameters (parameters m d)) := rfl

/-- Every input FP16 payload, including NaN payloads and signed zero, is copied
unchanged to its transposed coordinate. No numerical premise is required. -/
theorem wrapper_correct (m d : Nat) (h : InputContract m d)
    (samples : Nat → Word) (initial : State Word) :
    ∃ final,
      runWrapper m d samples initial =
        .ok (some (returnedTensor m d, final)) ∧
      (∀ row : Fin m, ∀ col : Fin d,
        final.mem "out" (col.val*m+row.val) = .mk .fp16 (formula samples d row.val col.val)) ∧
      (∀ o, o < m*d → final.mem "matrix" o = .mk .fp16 (samples o)) ∧
      (∀ (r : RegionName) o, (r ≠ "matrix" ∨ m*d ≤ o) → (r ≠ "out" ∨ m*d ≤ o) →
        final.mem r o = initial.mem r o) := by
  obtain ⟨final, he, _, _, hv, hi, hf⟩ :=
    Host.matrix_transpose_host_correctness (parameters m d) initial samples
      (by change ("matrix" : RegionName) ≠ "out"; decide)
  refine ⟨final, ?_, hv, hi, hf⟩
  simp [runWrapper, (check_nat_iff m d).2 h, he]

/-- Floating primitive implementations are unreachable throughout this source. -/
theorem backend_independent (m d : Nat) (initial : State Word)
    (left right : Algebra Word) :
    exec left (source "matrix" "out" m d) initial =
      exec right (source "matrix" "out" m d) initial := by
  rw [source_is_original]
  exact FPBits.backend_independent left right Bool.false _ initial

#axiomsClean source_is_original
#axiomsClean wrapper_correct
#axiomsClean backend_independent
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixTranspose.FP
