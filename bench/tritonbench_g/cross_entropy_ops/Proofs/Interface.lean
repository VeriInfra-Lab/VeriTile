import bench.tritonbench_g.cross_entropy_ops.IO
import bench.tritonbench_g.cross_entropy_ops.Proofs.AccessSafety
import bench.tritonbench_g.cross_entropy_ops.Proofs.LogicalExecution
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Interface
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSectionVars false
variable {A : Type} [Inhabited A]

@[simp] theorem invocation_mem (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (backward : Bool) (row block : Nat) (s : State A) :
    ((invocation M p v backward row block).initial s).mem = s.mem := Invocation.initial_mem _ _
@[simp] theorem invocation_row (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (backward : Bool) (row block : Nat) (s : State A) :
    ((invocation M p v backward row block).initial s).pids 0 = row := by
  simp [invocation, Invocation.initial, bindRegisters, arguments, State.setReg]
@[simp] theorem invocation_block (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (backward : Bool) (row block : Nat) (s : State A) :
    ((invocation M p v backward row block).initial s).pids 1 = block := by
  simp [invocation, Invocation.initial, bindRegisters, arguments, State.setReg]
theorem invocation_abi (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (backward : Bool) (row block : Nat) (s : State A) :
    RuntimeArgs.Bound p ((invocation M p v backward row block).initial s) v := by
  constructor
  · constructor <;> simp [invocation, Invocation.initial, bindRegisters, arguments, State.setReg]
  · simp [invocation, Invocation.initial, bindRegisters, arguments, State.setReg]

namespace Forward
theorem input_values (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (row block : Nat) (values : Inputs A p) (s : State A)
    (hin : (programIO M p v row block).InputBinding values s) :
    (∀ i, i < p.cols → (s.mem p.logits (p.logitsBase + row * p.logitsStride + i)).read .real = extend values i) ∧
      (s.mem (Region.cast p.labels) (p.labelsBase + row)).read .int = label values := by
  constructor
  · intro i hi
    simpa [programIO, memory, inputs, read, mapValue, extend, hi, logit] using
      hin ⟨0, by simp [programIO, memory, inputs]⟩ (⟨i, hi⟩, PUnit.unit) trivial
  · exact hin ⟨1, by simp [programIO, memory, inputs]⟩ PUnit.unit trivial

theorem safe (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (row block : Nat) (positive : 0 < p.block) (values : Inputs A p)
    (bounds : RegionName → Nat) (hb : (programIO M p v row block).InBounds bounds values)
    (s : State A) : ScheduleSafe bounds ((programIO M p v row block).schedule values) s := by
  apply (scheduleSafe_singleton _ _ _).mpr
  apply KernelNative.forward_traceSafe M p _ bounds positive v (invocation_abi M p v Bool.false row block s)
  · simpa using hb.2.1 ⟨1, by simp [programIO, memory, inputs]⟩ PUnit.unit trivial
  · intro i hi
    simpa using hb.2.1 ⟨0, by simp [programIO, memory, inputs]⟩ (⟨i, hi⟩, PUnit.unit) trivial
  · simpa using hb.2.2.1 ⟨0, by simp [programIO, memory, outputs]⟩ PUnit.unit (Or.inl (by simp))
  · simpa using hb.2.2.1 ⟨1, by simp [programIO, memory, outputs]⟩ PUnit.unit (Or.inl (by simp))
  · intro sp
    simpa using hb.2.2.1 ⟨2, by simp [programIO, memory, outputs]⟩ PUnit.unit (Or.inr sp)

theorem implements (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (row block : Nat) (positive : 0 < p.block)
    (separate : p.lse ≠ p.logits) (outputs : p.lse ≠ p.loss)
    (zLse : p.zloss ≠ p.lse) (zLoss : p.zloss ≠ p.loss) :
    (programIO M p v row block).Implements (fun values => result p
      (Ordered.lse M p v block (extend values))
      (Ordered.loss M p v block (extend values) (label values))
      (Ordered.zloss M p v block (extend values) (label values))) := by
  intro values _ bounds hb s hin
  refine ⟨safe M p v row block positive values bounds hb s, ?_⟩
  let initial := (invocation M p v Bool.false row block).initial s
  have abi := invocation_abi M p v Bool.false row block s
  obtain ⟨inputs, labels⟩ := input_values M p v row block values s hin
  have input : ∀ i, i < p.cols → (initial.mem p.logits
      (p.logitsBase + initial.pids 0 * p.logitsStride + i)).read .real = extend values i := by
    simpa [initial] using inputs
  have target : KernelNative.target p initial = label values := by
    simpa [KernelNative.target, initial] using labels
  obtain ⟨final, ran, lse, loss, zloss, frame⟩ :=
    KernelNative.forward_cells M p initial positive separate outputs zLse zLoss v abi
  rw [KernelNative.lse_logical M p initial v abi _ input] at lse
  rw [KernelNative.loss_logical M p initial v abi _ input, target] at loss
  rw [KernelNative.zloss_logical M p initial v abi _ input, target] at zloss
  simp only [initial, invocation_row, invocation_block, invocation_mem] at lse loss zloss frame
  refine ⟨final, ?_, ?_, ?_⟩
  · simpa only [programIO, runSchedule_singleton, Invocation.run] using ran
  · intro o j active
    fin_cases o
    · cases j
      change mapCell id (final.mem p.lse (p.lseBase + block * p.rows + row)) =
        .mk .real (Ordered.lse M p v block (extend values))
      rw [lse]
      rfl
    · cases j
      change mapCell id (final.mem p.loss (p.lossBase + block * p.rows + row)) =
        .mk .real (Ordered.loss M p v block (extend values) (label values))
      rw [loss]
      rfl
    · cases j
      have sp : p.split = Bool.false := active.resolve_left (by simp)
      change mapCell id (final.mem p.zloss (p.zlossBase + block * p.rows + row)) =
        .mk .real (Ordered.zloss M p v block (extend values) (label values))
      rw [zloss sp]
      rfl
  · intro r i outside
    apply frame r i
    · exact outside.1 ⟨1, by simp [programIO, memory, Interface.Forward.outputs]⟩ PUnit.unit (Or.inl (by simp))
    · exact outside.1 ⟨0, by simp [programIO, memory, Interface.Forward.outputs]⟩ PUnit.unit (Or.inl (by simp))
    · cases sp : p.split
      · exact Or.inr (outside.1 ⟨2, by simp [programIO, memory, Interface.Forward.outputs]⟩ PUnit.unit (Or.inr sp))
      · exact Or.inl rfl
end Forward

namespace Backward
theorem input_values (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (row block : Nat) (values : Inputs A p) (s : State A)
    (hin : (programIO M p v row block).InputBinding values s) :
    (∀ i, i < p.cols → (s.mem p.logits (p.logitsBase + row * p.logitsStride + i)).read .real = extend values i) ∧
      (s.mem (Region.cast p.labels) (p.labelsBase + row)).read .int = label values ∧
      (s.mem p.lse (p.lseBase + row)).read .real = lse values ∧
      (s.mem p.upstream (p.upstreamBase + row * p.upstreamStride)).read .real = upstream values := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i hi
    simpa [programIO, memory, inputs, read, mapValue, extend, hi, logit] using
      hin ⟨0, by simp [programIO, memory, inputs]⟩ (⟨i, hi⟩, PUnit.unit) trivial
  · exact hin ⟨1, by simp [programIO, memory, inputs]⟩ PUnit.unit trivial
  · exact hin ⟨2, by simp [programIO, memory, inputs]⟩ PUnit.unit trivial
  · exact hin ⟨3, by simp [programIO, memory, inputs]⟩ PUnit.unit trivial

theorem safe (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (row block : Nat) (values : Inputs A p)
    (bounds : RegionName → Nat) (hb : (programIO M p v row block).InBounds bounds values)
    (s : State A) : ScheduleSafe bounds ((programIO M p v row block).schedule values) s := by
  apply (scheduleSafe_singleton _ _ _).mpr
  apply KernelNative.backward_traceSafe M p _ bounds v (invocation_abi M p v Bool.true row block s)
  · simpa using hb.2.1 ⟨1, by simp [programIO, memory, inputs]⟩ PUnit.unit trivial
  · intro i hi
    simpa using hb.2.1 ⟨0, by simp [programIO, memory, inputs]⟩ (⟨i, hi⟩, PUnit.unit) trivial
  · simpa using hb.2.1 ⟨2, by simp [programIO, memory, inputs]⟩ PUnit.unit trivial
  · intro _
    simpa using hb.2.1 ⟨3, by simp [programIO, memory, inputs]⟩ PUnit.unit trivial
  · intro i active
    simpa using hb.2.2.1 ⟨0, by simp [programIO, memory, outputs]⟩ (i, PUnit.unit)
      (by simpa using active)

theorem implements (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (row block : Nat) :
    (programIO M p v row block).Implements (fun values => result p (fun i =>
      Ordered.gradient M p v block (extend values) (label values) (lse values) (upstream values) i)) := by
  intro values _ bounds hb s hin
  refine ⟨safe M p v row block values bounds hb s, ?_⟩
  let initial := (invocation M p v Bool.true row block).initial s
  have abi := invocation_abi M p v Bool.true row block s
  obtain ⟨inputs, labels, lseread, upstreamread⟩ := input_values M p v row block values s hin
  have input : ∀ i, i < p.cols → (initial.mem p.logits
      (p.logitsBase + initial.pids 0 * p.logitsStride + i)).read .real = extend values i := by
    simpa [initial] using inputs
  have target : KernelNative.target p initial = label values := by
    simpa [KernelNative.target, initial] using labels
  obtain ⟨final, ran, valuesRead, frame⟩ := KernelNative.backward_cells M p initial v abi
  simp only [KernelNative.gradientOffset, initial, invocation_row, invocation_block, invocation_mem] at frame
  refine ⟨final, ?_, ?_, ?_⟩
  · simpa only [programIO, runSchedule_singleton, Invocation.run] using ran
  · intro o j active
    fin_cases o
    have active' : initial.pids 1 * p.block + j.1.val < p.cols := by simpa [initial] using active
    have cell := valuesRead j.1 active'
    rw [KernelNative.gradient_logical M p initial v abi _ input, target] at cell
    simp only [KernelNative.gradientOffset, initial, invocation_row, invocation_block,
      invocation_mem, lseread, upstreamread] at cell
    change mapCell id (final.mem p.gradient (p.gradientBase + row * p.gradientStride + (block * p.block + j.1.val))) =
      .mk .real (Ordered.gradient M p v block (extend values) (label values) (lse values) (upstream values) j.1)
    rw [cell]
    rfl
  · intro r i outside
    apply frame r i
    intro lane active
    exact outside.1 ⟨0, by simp [programIO, memory, outputs]⟩ (lane, PUnit.unit) active
end Backward

#axiomsClean Forward.implements
#axiomsClean Backward.implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Interface
