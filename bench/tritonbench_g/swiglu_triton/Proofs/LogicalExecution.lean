import bench.tritonbench_g.swiglu_triton.Proofs.DirectTile
import bench.tritonbench_g.swiglu_triton.Proofs.Math
import bench.tritonbench_g.swiglu_triton.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.SwigluTriton.LogicalExecution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term graphAlgebra)
open Native
noncomputable section

private theorem direct_scalar_real (ch : Channel) (a b dc : ℝ) :
    DirectTile.scalar Math.algebra ch a b dc = Math.formula ch a b dc :=
  Math.scalar_real ch a b dc

/-- Active logical input channels suffice for the complete actual source.
The frame is the exact typed-cell frame, including disabled output channels. -/
theorem inputs {α : Type} [Inhabited α] (M : Algebra α) (backward : Bool)
    (block cols row : Nat) (s : State α) (a b dc : Fin block → α)
    (ha : ∀ j : Fin block, j.val < cols →
      (s.mem "A" (row*cols+j.val)).read .real = a j)
    (hb : ∀ j : Fin block, j.val < cols →
      (s.mem "B" (row*cols+j.val)).read .real = b j)
    (hdc : backward = Bool.true → ∀ j : Fin block, j.val < cols →
      (s.mem "DC" (row*cols+j.val)).read .real = dc j) :
    ∃ final, exec M (directSource backward block cols) (DirectTile.rowState row s) = some final ∧
      (∀ ch, enabled backward ch → ∀ j : Fin block, j.val < cols →
        final.mem (outputRegion ch) (row*cols+j.val) =
          Cell.mk .real (DirectTile.scalar M ch (a j) (b j) (dc j))) ∧
      (∀ r o, DirectTile.Outside backward block cols row r o → final.mem r o = s.mem r o) := by
  refine ⟨(DirectTile.sourceRun M backward block cols row s).val,
    (DirectTile.sourceRun M backward block cols row s).property, ?_,
    DirectTile.frame M backward block cols row s⟩
  intro ch on j active
  have value := DirectTile.readback M backward block cols row s ch on (j, PUnit.unit) active
  rw [ha j active, hb j active] at value
  cases backward
  · have same : ch = .c := on
    subst ch
    exact value
  · rw [hdc rfl j active] at value
    exact value

/-- The mathematical channel formula is independent of source execution. -/
theorem real_inputs (backward : Bool) (block cols row : Nat) (s : State ℝ)
    (a b dc : Fin block → ℝ)
    (ha : ∀ j : Fin block, j.val < cols → (s.mem "A" (row*cols+j.val)).read .real = a j)
    (hb : ∀ j : Fin block, j.val < cols → (s.mem "B" (row*cols+j.val)).read .real = b j)
    (hdc : backward = Bool.true → ∀ j : Fin block, j.val < cols →
      (s.mem "DC" (row*cols+j.val)).read .real = dc j) :
    ∃ final, exec Math.algebra (directSource backward block cols) (DirectTile.rowState row s) = some final ∧
      (∀ ch, enabled backward ch → ∀ j : Fin block, j.val < cols →
        final.mem (outputRegion ch) (row*cols+j.val) =
          Cell.mk .real (Math.formula ch (a j) (b j) (dc j))) ∧
      (∀ r o, DirectTile.Outside backward block cols row r o → final.mem r o = s.mem r o) := by
  simpa only [direct_scalar_real] using inputs Math.algebra backward block cols row s a b dc ha hb hdc

/-- Native observations use raw FP32 logical words at the active loads.
No finite-word restriction or arithmetic transformation law is required. -/
theorem native_inputs (instructions : InstructionSemantics) (backward : Bool)
    (block cols row : Nat) (s : State (Term Word)) (a b dc : Fin block → Word)
    (ha : ∀ j : Fin block, j.val < cols →
      (s.mem "A" (row*cols+j.val)).read .real = .input (a j))
    (hb : ∀ j : Fin block, j.val < cols →
      (s.mem "B" (row*cols+j.val)).read .real = .input (b j))
    (hdc : backward = Bool.true → ∀ j : Fin block, j.val < cols →
      (s.mem "DC" (row*cols+j.val)).read .real = .input (dc j)) :
    ∃ final, exec graphAlgebra (directSource backward block cols) (DirectTile.rowState row s) = some final ∧
      (∀ ch, enabled backward ch → ∀ j : Fin block, j.val < cols →
        interpret (compiledPorts instructions)
          ((final.mem (outputRegion ch) (row*cols+j.val)).read .real) =
            compiledReference instructions ch (a j) (b j) (dc j)) ∧
      (∀ r o, DirectTile.Outside backward block cols row r o → final.mem r o = s.mem r o) := by
  obtain ⟨final, executed, values, frame⟩ := inputs graphAlgebra backward block cols row s
    (fun j => .input (a j)) (fun j => .input (b j)) (fun j => .input (dc j)) ha hb hdc
  refine ⟨final, executed, ?_, frame⟩
  intro ch on j active
  rw [values ch on j active, Cell.read_same]
  exact KernelNative.scalar_lowering instructions ch (a j) (b j) (dc j)

/-- The retained raw-state specialization used by the optional Tensor bridge.
Execution and values come from the same logical-input foundation as public IO. -/
theorem real_kernel (backward : Bool) (block cols row : Nat)
    (covers : cols ≤ block) (entry : State ℝ) :
    VeriTile.Spec.Real (∃ final, exec Math.algebra (directSource backward block cols)
        (DirectTile.rowState row entry) = some final ∧
      (∀ ch, enabled backward ch → ∀ col, col < cols →
        final.mem (outputRegion ch) (row*cols+col) = Cell.mk .real
          (Math.formula ch ((entry.mem "A" (row*cols+col)).read .real)
            ((entry.mem "B" (row*cols+col)).read .real)
            ((entry.mem "DC" (row*cols+col)).read .real))) ∧
      (∀ r o, DirectTile.Outside backward block cols row r o → final.mem r o = entry.mem r o)) := by
  obtain ⟨final, executed, values, frame⟩ := real_inputs backward block cols row entry
    (fun j => (entry.mem "A" (row*cols+j.val)).read .real)
    (fun j => (entry.mem "B" (row*cols+j.val)).read .real)
    (fun j => (entry.mem "DC" (row*cols+j.val)).read .real)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ _ => rfl)
  refine ⟨final, executed, ?_, frame⟩
  intro ch on col active
  exact values ch on ⟨col, lt_of_lt_of_le active covers⟩ active

#axiomsClean real_kernel
#axiomsClean inputs
#axiomsClean real_inputs
#axiomsClean native_inputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.LogicalExecution
