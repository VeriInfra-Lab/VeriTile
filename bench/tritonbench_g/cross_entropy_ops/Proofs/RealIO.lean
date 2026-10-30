import bench.tritonbench_g.cross_entropy_ops.Proofs.Interface
import bench.tritonbench_g.cross_entropy_ops.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Interface
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
set_option maxHeartbeats 4000000

theorem mapCell_id (c : Cell A) : mapCell id c = c := by
  cases c with | mk d v => cases d <;> rfl
theorem read_replace [Inhabited A] (c : Cell A) (a b : A)
    (typed : mapCell id c = .mk .real a) (value : c.read .real = b) :
    mapCell id c = .mk .real b := by
  rw [mapCell_id] at typed ⊢
  rw [typed] at value ⊢
  simpa using congrArg (Cell.mk .real) value

namespace Forward
theorem real_input (p : Parameters) (values : Inputs (WithBot ℝ) p)
    (domain : RealInputs p values) (i : Nat) (hi : i < p.cols) :
    extend values i = (realInput values i : WithBot ℝ) := by
  obtain ⟨x, hx⟩ := domain ⟨i, hi⟩
  simp [extend, realInput, hi, hx]
  rfl

theorem real_unsplit (p : Parameters) (row : Nat)
    (rowBound : row < p.rows) (positive : 0 < p.cols) (covered : p.cols ≤ p.block)
    (split : p.split = Bool.false) (noSmooth : p.hasSmoothing = Bool.false → p.smoothing = 0)
    (start : p.classStart = 0) (total : p.totalClasses = p.cols)
    (separate : p.lse ≠ p.logits) (outputs : p.lse ≠ p.loss)
    (zLse : p.zloss ≠ p.lse) (zLoss : p.zloss ≠ p.loss) :
    (programIO KernelReal.algebra p (KernelReal.arguments p) row 0).ImplementsOn
      (fun values => RealInputs p values ∧
        (label values = p.ignored ∨ (0 ≤ label values ∧ label values < (p.cols : Int))))
      (fun values => result p
        (Spec.lse p.cols p.logitScale (realInput values) : WithBot ℝ)
        (Spec.loss p.cols p.logitScale p.smoothing p.squareScale (realInput values) (label values) p.ignored : WithBot ℝ)
        (Spec.zLoss p.cols p.logitScale p.squareScale (realInput values) (label values) p.ignored : WithBot ℝ)) := by
  intro values domain bounds hb s hin
  obtain ⟨safe, final, ran, observed, frame⟩ :=
    implements KernelReal.algebra p (KernelReal.arguments p) row 0 (by omega)
      separate outputs zLse zLoss values trivial bounds hb s hin
  let initial := (invocation KernelReal.algebra p (KernelReal.arguments p) Bool.false row 0).initial s
  have abi := invocation_abi KernelReal.algebra p (KernelReal.arguments p) Bool.false row 0 s
  obtain ⟨input, labels⟩ := input_values KernelReal.algebra p (KernelReal.arguments p) row 0 values s hin
  have input' : RealSource.Input p (scalarMemory initial) row (realInput values) := by
    intro i hi
    change (initial.mem p.logits (p.logitsBase + row * p.logitsStride + i)).read .real = _
    simpa [initial] using (input i hi).trans (real_input p values domain.1 i hi)
  obtain ⟨other, run', lse, loss, zloss, _⟩ := RawKernel.forward_unsplit p separate outputs zLse zLoss
    initial abi row rowBound (invocation_row _ _ _ _ _ _ _) (invocation_block _ _ _ _ _ _ _)
    (realInput values) input' (label values) (by simpa [CrossEntropy1.Source.label,
      scalarMemory, CrossEntropy1.KernelContract.scalarMemory, CrossEntropy1.Source.Cell.read, initial] using labels)
    positive covered split noSmooth start total domain.2
  have ran' : exec (KernelNative.engine KernelReal.algebra) (forwardSyntax p) initial = some final := by
    simpa only [programIO, runSchedule_singleton, Invocation.run] using ran
  have same : other = final := Option.some.inj (run'.symm.trans ran')
  subst other
  refine ⟨safe, final, ran, ?_, frame⟩
  intro o j active
  have typed := observed o j active
  fin_cases o
  · cases j
    exact read_replace _ _ _ typed (by simpa [programIO, memory, outputs, write, result] using lse)
  · cases j
    exact read_replace _ _ _ typed (by simpa [programIO, memory, outputs, write, result] using loss)
  · cases j
    exact read_replace _ _ _ typed (by simpa [programIO, memory, outputs, write, result] using zloss)

theorem real_split (p : Parameters) (row block : Nat)
    (rowBound : row < p.rows) (positive : 0 < p.block) (active : block * p.block < p.cols)
    (split : p.split = Bool.true) (noSmooth : p.hasSmoothing = Bool.false → p.smoothing = 0)
    (separate : p.lse ≠ p.logits) (outputs : p.lse ≠ p.loss)
    (zLse : p.zloss ≠ p.lse) (zLoss : p.zloss ≠ p.loss) :
    (programIO KernelReal.algebra p (KernelReal.arguments p) row block).ImplementsOn
      (RealInputs p) (fun values => result p
        (CrossEntropy1.Math.blockLSE p.cols p.block block (Spec.scaled p.logitScale (realInput values)) : WithBot ℝ)
        (CrossEntropy1.RealSource.partialLoss p.toParameters (Spec.scaled p.logitScale (realInput values))
          (label values) block : WithBot ℝ) (0 : ℝ)) := by
  intro values domain bounds hb s hin
  obtain ⟨safe, final, ran, observed, frame⟩ :=
    implements KernelReal.algebra p (KernelReal.arguments p) row block positive
      separate outputs zLse zLoss values trivial bounds hb s hin
  let initial := (invocation KernelReal.algebra p (KernelReal.arguments p) Bool.false row block).initial s
  have abi := invocation_abi KernelReal.algebra p (KernelReal.arguments p) Bool.false row block s
  obtain ⟨input, labels⟩ := input_values KernelReal.algebra p (KernelReal.arguments p) row block values s hin
  have input' : RealSource.Input p (scalarMemory initial) row (realInput values) := by
    intro i hi
    change (initial.mem p.logits (p.logitsBase + row * p.logitsStride + i)).read .real = _
    simpa [initial] using (input i hi).trans (real_input p values domain i hi)
  obtain ⟨other, run', lse, loss, _⟩ := RawKernel.forward_split p separate outputs zLse zLoss
    initial abi row block rowBound (invocation_row _ _ _ _ _ _ _) (invocation_block _ _ _ _ _ _ _)
    (realInput values) input' (label values) (by simpa [CrossEntropy1.Source.label,
      scalarMemory, CrossEntropy1.KernelContract.scalarMemory, CrossEntropy1.Source.Cell.read, initial] using labels)
    positive active split noSmooth
  have ran' : exec (KernelNative.engine KernelReal.algebra) (forwardSyntax p) initial = some final := by
    simpa only [programIO, runSchedule_singleton, Invocation.run] using ran
  have same : other = final := Option.some.inj (run'.symm.trans ran')
  subst other
  refine ⟨safe, final, ran, ?_, frame⟩
  intro o j activeOutput
  have typed := observed o j activeOutput
  fin_cases o
  · cases j
    exact read_replace _ _ _ typed lse
  · cases j
    exact read_replace _ _ _ typed loss
  · exfalso
    simpa [programIO, memory, split] using activeOutput
end Forward

namespace Backward
theorem real_input (p : Parameters) (values : Inputs (WithBot ℝ) p)
    (domain : RealInputs p values) (i : Nat) (hi : i < p.cols) :
    extend values i = (realInput values i : WithBot ℝ) := by
  obtain ⟨x, hx⟩ := domain.1 ⟨i, hi⟩
  simp [extend, realInput, hi, hx]
  rfl
theorem real_implements (p : Parameters) (row block : Nat) (rowBound : row < p.rows)
    (lseLogits : p.lse ≠ p.logits) (lseUpstream : p.lse ≠ p.upstream)
    (noSmooth : p.hasSmoothing = Bool.false → p.smoothing = 0) :
    (programIO KernelReal.algebra p (KernelReal.arguments p) row block).ImplementsOn
      (RealInputs p) (fun values => result p (fun i =>
        (RawKernel.gradient p (realInput values) (label values)
          ((lse values).getD 0) ((upstream values).getD 0) (block * p.block + i.val) : WithBot ℝ))) := by
  intro values domain bounds hb s hin
  obtain ⟨safe, final, ran, observed, frame⟩ :=
    implements KernelReal.algebra p (KernelReal.arguments p) row block values trivial bounds hb s hin
  let initial := (invocation KernelReal.algebra p (KernelReal.arguments p) Bool.true row block).initial s
  have abi := invocation_abi KernelReal.algebra p (KernelReal.arguments p) Bool.true row block s
  obtain ⟨input, labels, lseRead, upstreamRead⟩ := input_values KernelReal.algebra p (KernelReal.arguments p) row block values s hin
  have input' : RealSource.Input p (scalarMemory initial) row (realInput values) := by
    intro i hi
    change (initial.mem p.logits (p.logitsBase + row * p.logitsStride + i)).read .real = _
    simpa [initial] using (input i hi).trans (real_input p values domain i hi)
  have lse' : lse values = ((lse values).getD 0 : WithBot ℝ) := by
    obtain ⟨x, hx⟩ := domain.2.1
    simp [hx]
    rfl
  have upstream' : upstream values = ((upstream values).getD 0 : WithBot ℝ) := by
    obtain ⟨x, hx⟩ := domain.2.2
    simp [hx]
    rfl
  obtain ⟨other, run', computed, _⟩ := RawKernel.backward p initial abi row block rowBound
    (invocation_row _ _ _ _ _ _ _) (invocation_block _ _ _ _ _ _ _)
    (realInput values) ((lse values).getD 0) ((upstream values).getD 0) input' (label values)
    (by simpa [CrossEntropy1.Source.label, scalarMemory, CrossEntropy1.KernelContract.scalarMemory,
      CrossEntropy1.Source.Cell.read, initial] using labels)
    (by simpa [scalarMemory, CrossEntropy1.KernelContract.scalarMemory, CrossEntropy1.Source.Cell.read, initial] using lseRead.trans lse')
    (by simpa [scalarMemory, CrossEntropy1.KernelContract.scalarMemory, CrossEntropy1.Source.Cell.read, initial] using upstreamRead.trans upstream')
    lseLogits lseUpstream noSmooth
  have ran' : exec (KernelNative.engine KernelReal.algebra) (backwardSyntax p) initial = some final := by
    simpa only [programIO, runSchedule_singleton, Invocation.run] using ran
  have same : other = final := Option.some.inj (run'.symm.trans ran')
  subst other
  refine ⟨safe, final, ran, ?_, frame⟩
  intro o j active
  have typed := observed o j active
  fin_cases o
  exact read_replace _ _ _ typed (computed j.1 active)
end Backward

#axiomsClean Forward.real_unsplit
#axiomsClean Forward.real_split
#axiomsClean Backward.real_implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Interface
