import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NativeIO

/-! The new logical channels specialize to every former raw-word entry, with
the exact same native expressions. This is an input/formula connection only;
the shared execution and source-safety proofs are reused unchanged. -/
namespace VeriTile.Bench.Repairs.CrossEntropyOps.Interface.Native
open VeriTile.Triton FP.Structural KernelProfile KernelContract
noncomputable section
set_option maxHeartbeats 4000000

def forwardWords (p : Parameters) (row : Nat) (words : State Word) : Forward.Inputs Word p :=
  fun i => match i with
  | ⟨0, _⟩ => fun j => (words.mem p.logits (p.logitsBase + row * p.logitsStride + j.1.val)).read .real
  | ⟨1, _⟩ => fun _ => (words.mem (Region.cast p.labels) (p.labelsBase + row)).read .int
  | ⟨n + 2, h⟩ => False.elim (by simp [Forward.inputs] at h)
def backwardWords (p : Parameters) (row : Nat) (words : State Word) : Backward.Inputs Word p :=
  fun i => match i with
  | ⟨0, _⟩ => fun j => (words.mem p.logits (p.logitsBase + row * p.logitsStride + j.1.val)).read .real
  | ⟨1, _⟩ => fun _ => (words.mem (Region.cast p.labels) (p.labelsBase + row)).read .int
  | ⟨2, _⟩ => fun _ => (words.mem p.lse (p.lseBase + row)).read .real
  | ⟨3, _⟩ => fun _ => (words.mem p.upstream (p.upstreamBase + row * p.upstreamStride)).read .real
  | ⟨n + 4, h⟩ => False.elim (by simp [Backward.inputs] at h)

theorem entry_fp (p : Parameters) (v : RuntimeArgs.Values Word) (words : State Word)
    (r : RegionName) (i : Nat) :
    ((entry p v words).mem r i).read .real = encode p r i ((words.mem r i).read .real) := by
  by_cases same : r = Region.cast p.labels
  · simp only [entry, State.setReg, same, if_pos, encode]
    rfl
  · simp only [entry, State.setReg, same, if_neg, encode]
    rfl

theorem entry_label (p : Parameters) (v : RuntimeArgs.Values Word) (words : State Word) (i : Nat) :
    ((entry p v words).mem (Region.cast p.labels) i).read .int =
      (words.mem (Region.cast p.labels) i).read .int := by
  simp only [entry, State.setReg, if_pos rfl]
  rfl

theorem forward_entry_inputs (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (v : RuntimeArgs.Values Word) (words : State Word) :
    (forwardIO R I S p v (words.pids 0) (words.pids 1)).InputBinding
      (forwardWords p (words.pids 0) words) (entry p v words) := by
  intro i j active
  fin_cases i
  · exact entry_fp p v words p.logits _
  · exact entry_label p v words _

theorem backward_entry_inputs (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (v : RuntimeArgs.Values Word) (words : State Word) :
    (backwardIO R I S p v (words.pids 0) (words.pids 1)).InputBinding
      (backwardWords p (words.pids 0) words) (entry p v words) := by
  intro i j active
  fin_cases i
  · exact entry_fp p v words p.logits _
  · exact entry_label p v words _
  · exact entry_fp p v words p.lse _
  · exact entry_fp p v words p.upstream _

theorem forward_old_formula (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (v : RuntimeArgs.Values Word) (words : State Word) :
    forwardResult R I S p v (words.pids 0) (words.pids 1) (forwardWords p (words.pids 0) words) =
      Forward.result p (Compiler.eval I S (forwardLSE R p v words))
        (Compiler.eval I S (forwardLoss R p v words)) (Compiler.eval I S (forwardZ R p v words)) := by
  let values := forwardWords p (words.pids 0) words
  let x := Forward.extend (forwardInputs p (words.pids 0) values)
  have input : ∀ i, i < p.cols → ((entry p v words).mem p.logits
      (p.logitsBase + (entry p v words).pids 0 * p.logitsStride + i)).read .real = x i := by
    intro i hi
    simpa [x, values, Forward.extend, hi, forwardInputs, Forward.logit, forwardWords,
      entry, State.setReg] using entry_fp p v words p.logits (p.logitsBase + words.pids 0 * p.logitsStride + i)
  have label : KernelNative.target p (entry p v words) = Forward.label values := by
    exact entry_label p v words _
  have lse := KernelNative.lse_logical (Compiler.algebra R) p (entry p v words)
    (KernelProfile.arguments v) (entry_binding p v words) x input
  have loss := KernelNative.loss_logical (Compiler.algebra R) p (entry p v words)
    (KernelProfile.arguments v) (entry_binding p v words) x input
  have zloss := KernelNative.zloss_logical (Compiler.algebra R) p (entry p v words)
    (KernelProfile.arguments v) (entry_binding p v words) x input
  have pid : (entry p v words).pids = words.pids := rfl
  simp only [label, pid] at lse loss zloss
  simp only [forwardResult, forwardLSE, forwardLoss, forwardZ]
  rw [lse, loss, zloss]

theorem backward_old_formula (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (v : RuntimeArgs.Values Word) (words : State Word) :
    backwardResult R I S p v (words.pids 0) (words.pids 1) (backwardWords p (words.pids 0) words) =
      Backward.result p (fun i => Compiler.eval I S (backwardLane R p v words i)) := by
  let values := backwardWords p (words.pids 0) words
  let data := backwardInputs p (words.pids 0) values
  let x := Backward.extend data
  have input : ∀ i, i < p.cols → ((entry p v words).mem p.logits
      (p.logitsBase + (entry p v words).pids 0 * p.logitsStride + i)).read .real = x i := by
    intro i hi
    simpa [x, data, values, Backward.extend, hi, backwardInputs, Backward.logit, backwardWords,
      entry, State.setReg] using entry_fp p v words p.logits (p.logitsBase + words.pids 0 * p.logitsStride + i)
  have label : KernelNative.target p (entry p v words) = Backward.label values := by
    exact entry_label p v words _
  have lseread : ((entry p v words).mem p.lse (p.lseBase + words.pids 0)).read .real = Backward.lse data :=
    entry_fp p v words p.lse _
  have upstreamread : ((entry p v words).mem p.upstream (p.upstreamBase + words.pids 0 * p.upstreamStride)).read .real =
      Backward.upstream data := entry_fp p v words p.upstream _
  have computed (i : Fin p.block) := KernelNative.gradient_logical (Compiler.algebra R) p (entry p v words)
    (KernelProfile.arguments v) (entry_binding p v words) x input i
  have pid : (entry p v words).pids = words.pids := rfl
  simp only [label, pid, lseread, upstreamread] at computed
  simp only [backwardResult, backwardLane]
  simp only [computed]
  rfl

#axiomsClean forward_entry_inputs
#axiomsClean backward_entry_inputs
#axiomsClean forward_old_formula
#axiomsClean backward_old_formula
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Interface.Native
