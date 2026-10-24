import bench.repairs.tritonbench_g.matrix_transpose.Kernels
import bench.repairs.tritonbench_g.matrix_transpose.Proofs.Indices
import bench.tritonbench_g.matrix_transpose.Proofs.Real
import bench.tritonbench_g.matrix_transpose.Proofs.Source

/-! Mathematical interpretation of the guarded wrapper under ordinary successful
allocation and launch. Samples are arbitrary values, not an RNG assumption.
Fresh allocations bind the two distinct logical region names below. -/
namespace VeriTile.Bench.Repairs.MatrixTranspose.Real
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.MatrixTranspose
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def allocated (m d : Nat) (samples : Nat → ℝ) (initial : BlockState) : BlockState :=
  {initial with mem := fun r o =>
    if r = "matrix" ∧ o < m*d then .real (samples o)
    else if r = "out" ∧ o < m*d then .real 0
    else initial.mem r o}

def launchState (s : BlockState) : BlockState :=
  {s with regs := fun _ _ _ => none, pids := fun _ => 0, numPids := fun _ => 1}

def runWrapper (m d : Nat) (samples : Nat → ℝ) (initial : BlockState) :
    Except InputError (Option (ResultMetadata × BlockState)) := do
  checkInputs (.integer m) (.integer d)
  let final ← pure (exec ((source "matrix" "out" m d).eraseDType.toAlgKernel)
    (launchState (allocated m d samples initial)))
  return final.map fun s => (returnedTensor m d, s)

theorem source_projection (m d : Nat) :
    ((source "matrix" "out" m d).eraseDType.toAlgKernel).body =
      (kernel "matrix" "out" (max d 1) 1 (max m 1) 1 m d).toAlgKernel.body := by
  unfold source
  rw [typed_projection]
  rfl

private theorem output_injective (m d : Nat) :
    Function.Injective (fun i : TileIndex [d,m] => outAddr (max m 1) 1 i) := by
  intro a b eq
  have ha : a.2.1.val < m := a.2.1.isLt
  have hb : b.2.1.val < m := b.2.1.isLt
  have hm : 1 ≤ m := by omega
  simp only [outAddr, Nat.max_eq_left hm, Nat.mul_one] at eq
  have lower := congrArg (fun o => o % m) eq
  dsimp only at lower
  rw [Nat.mul_add_mod_of_lt ha, Nat.mul_add_mod_of_lt hb] at lower
  have upper : a.1.val = b.1.val :=
    Nat.mul_right_cancel (by omega : 0 < m) (by omega)
  apply Prod.ext (Fin.ext upper)
  exact Prod.ext (Fin.ext lower) (Subsingleton.elim _ _)

/-- Complete mathematical output, input preservation, and exterior frame. The
source supplies every output value; allocation supplies only its own cells. -/
theorem wrapper_correct (m d : Nat) (h : InputContract m d)
    (samples : Nat → ℝ) (initial : BlockState) :
    ∃ final,
      runWrapper m d samples initial =
        .ok (some (returnedTensor m d, final)) ∧
      (∀ row : Fin m, ∀ col : Fin d,
        final.readMem "out" (col.val*m+row.val) = formula samples d row.val col.val) ∧
      (∀ o, o < m*d → final.readMem "matrix" o = samples o) ∧
      (∀ (r : RegionName) o, r ≠ "out" ∨ m*d ≤ o →
        final.mem r o = (allocated m d samples initial).mem r o) ∧
      (∀ (r : RegionName) o, (r ≠ "matrix" ∨ m*d ≤ o) → (r ≠ "out" ∨ m*d ≤ o) →
        final.mem r o = initial.mem r o) := by
  let s := launchState (allocated m d samples initial)
  let xs : Fin (d*m) → ℝ := fun j =>
    s.readMem "matrix" (matrixAddr (max d 1) 1 (Lane2D.decode j))
  have inj : Function.Injective (fun i : TileIndex [d,m] => outAddr (max m 1) 1 i) := by
    exact output_injective m d
  obtain ⟨final, he, hv, hf⟩ :=
    kernel_region_run "matrix" "out" (max d 1) 1 (max m 1) 1 m d s xs inj (fun _ => rfl)
  have execution : exec ((source "matrix" "out" m d).eraseDType.toAlgKernel) s = some final := by
    simpa only [exec, source_projection] using he
  have frame (r : RegionName) (o : Nat) (outside : r ≠ "out" ∨ m*d ≤ o) :
      final.mem r o = (allocated m d samples initial).mem r o := by
    apply hf r o
    rcases outside with hn | he
    · exact Or.inl hn
    · right
      intro j eq
      have bound : outAddr (max m 1) 1 (Lane2D.decode j) < m*d := by
        simpa [outAddr, Nat.max_eq_left h.positive.1, Nat.mul_comm] using
          Indices.flat_bound d m (Lane2D.decode j).1.val (Lane2D.decode j).2.1.val
            (Lane2D.decode j).1.isLt (Lane2D.decode j).2.1.isLt
      omega
  refine ⟨final, ?_, ?_, ?_, frame, ?_⟩
  · simp [runWrapper, (check_nat_iff m d).2 h, execution, s]
  · intro row col
    have hm := h.positive.1
    have hd := h.positive.2
    have bound := Indices.flat_bound m d row.val col.val row.isLt col.isLt
    have value := hv (Lane2D.encode (col,row,()))
    simpa [xs, Lane2D.decode_encode, outAddr, matrixAddr, Nat.max_eq_left hm,
      Nat.max_eq_left hd, s, launchState, allocated, BlockState.readMem, bound, formula] using value
  · intro o ho
    have value := frame "matrix" o (Or.inl (by decide))
    simp only [BlockState.readMem, value]
    simp [allocated, ho]
  · intro r o hi ho
    rw [frame r o ho]
    have ni : ¬(r = "matrix" ∧ o < m*d) := by rcases hi with hn | he <;> simp_all
    have no : ¬(r = "out" ∧ o < m*d) := by rcases ho with hn | he <;> simp_all
    simp [allocated, ni, no]

#axiomsClean source_projection
#axiomsClean wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixTranspose.Real
