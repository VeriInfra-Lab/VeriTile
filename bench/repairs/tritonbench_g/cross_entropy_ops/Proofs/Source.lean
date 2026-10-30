import bench.repairs.tritonbench_g.cross_entropy_ops.Kernels
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Execution

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Source
open VeriTile.Triton
open CrossEntropy1.Source (FP32 Memory Cell write fp)
noncomputable section

/-- FP32 input words already have their source dtype; the explicit FP32 cast
is the identity. Every arithmetic operation, including both scale products,
remains in the expression graph. -/
def arithmetic {α : Type} (M : FP32 α) : FP32 α := {M with load := id}

/-- A read-only expression substitution, not a Torch operation or a stored
scaled tensor. It makes CE1's unchanged arithmetic helpers reusable. Host
preparation supplies separate labels/upstream snapshots; output may alias logits.
Only the real channel changes. Pointer coordinates and integer labels do not. -/
def scaledReads {α : Type} (M : FP32 α) (p : Parameters) (memory : Memory α)
    (backward : Bool) : Memory α := fun region offset =>
  ⟨fun dtype => match dtype with
    | .real =>
      if region=p.logits ∨ (backward=Bool.true ∧ region=p.upstream) then
        M.mul (M.load ((memory region offset).read .real)) (M.literal p.logitScale)
      else M.load ((memory region offset).read .real)
    | dtype => (memory region offset).read dtype⟩

def forwardValues {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) : CrossEntropy1.Source.ForwardValues α p.block :=
  CrossEntropy1.Source.forwardValues (arithmetic M) p.toParameters (scaledReads M p memory Bool.false) row block

def forwardLoss {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (afterLSE : Memory α) (row block : Nat) (target : Int)
    (values : CrossEntropy1.Source.ForwardValues α p.block) : α :=
  CrossEntropy1.Source.forwardLoss (arithmetic M) p.toParameters (scaledReads M p afterLSE Bool.false)
    row block target values

def zLoss {α : Type} (M : FP32 α) (p : Parameters) (target : Int) (lse : α) : α :=
  if target=p.ignored then fp M 0 else
    if p.split then fp M 0 else M.mul (M.mul (fp M p.squareScale) lse) lse

/-- Actual order: labels and logits reads, LSE store, conditional target gather,
loss store, then non-split logging z-loss store. -/
def forward {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) : Memory α :=
  let target := CrossEntropy1.Source.label p.toParameters memory row
  let values := forwardValues M p memory row block
  let lseWritten := write memory p.lse (p.lseBase+block*p.rows+row) (.mk .real values.lse)
  let loss := forwardLoss M p lseWritten row block target values
  let lossWritten := write lseWritten p.loss (p.lossBase+block*p.rows+row) (.mk .real loss)
  if p.split then lossWritten else
    write lossWritten p.zloss (p.zlossBase+block*p.rows+row) (.mk .real (zLoss M p target values.lse))

def backwardValues {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) : Fin p.block → α :=
  CrossEntropy1.Source.backwardValues (arithmetic M) p.toParameters (scaledReads M p memory Bool.true) row block

/-- Every read and arithmetic operand precedes the single masked vector store.
The explicit ignored select remains after the probability arithmetic. -/
def backward {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) : Memory α :=
  let values := backwardValues M p memory row block
  (List.ofFn (fun lane : Fin p.block => lane)).foldl (fun current lane =>
    if CrossEntropy1.Source.column p.toParameters block lane.val<p.cols then
      write current p.gradient
        (CrossEntropy1.Source.gradientAddress p.toParameters row
          (CrossEntropy1.Source.column p.toParameters block lane.val)) (.mk .real (values lane))
    else current) memory

theorem scaled_integer_channel {α : Type} (M : FP32 α) (p : Parameters)
    (memory : Memory α) (backward : Bool) (region : RegionName) (offset : Nat) :
    ((scaledReads M p memory backward) region offset).read .int=(memory region offset).read .int := rfl

theorem label_unchanged {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (backward : Bool) (row : Nat) :
    CrossEntropy1.Source.label p.toParameters (scaledReads M p memory backward) row=
      CrossEntropy1.Source.label p.toParameters memory row := rfl

theorem backward_ignored {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat)
    (ignored : CrossEntropy1.Source.label p.toParameters memory row=p.ignored) (lane : Fin p.block) :
    backwardValues M p memory row block lane=fp M 0 := by
  exact CrossEntropy1.Program.backward_ignored (arithmetic M) p.toParameters
    (scaledReads M p memory Bool.true) row block ignored lane

theorem forward_frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) (region : RegionName) (offset : Nat)
    (loss : region≠p.loss ∨ offset≠p.lossBase+block*p.rows+row)
    (lse : region≠p.lse ∨ offset≠p.lseBase+block*p.rows+row)
    (zloss : region≠p.zloss ∨ offset≠p.zlossBase+block*p.rows+row) :
    forward M p memory row block region offset=memory region offset := by
  unfold forward
  split
  · rw [CrossEntropy1.Program.write_other _ _ _ _ _ _ loss,
      CrossEntropy1.Program.write_other _ _ _ _ _ _ lse]
  · rw [CrossEntropy1.Program.write_other _ _ _ _ _ _ zloss,
      CrossEntropy1.Program.write_other _ _ _ _ _ _ loss,
      CrossEntropy1.Program.write_other _ _ _ _ _ _ lse]

theorem backward_output {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) (lane : Fin p.block)
    (active : CrossEntropy1.Source.column p.toParameters block lane.val<p.cols) :
    (backward M p memory row block p.gradient
      (CrossEntropy1.Source.gradientAddress p.toParameters row
        (CrossEntropy1.Source.column p.toParameters block lane.val))).read .real=
      backwardValues M p memory row block lane := by
  have unique : Function.Injective (fun i : Fin p.block =>
      CrossEntropy1.Source.gradientAddress p.toParameters row
        (CrossEntropy1.Source.column p.toParameters block i.val)) := by
    intro i j h
    apply Fin.ext
    dsimp [CrossEntropy1.Source.gradientAddress,CrossEntropy1.Source.column] at h
    omega
  have h := CrossEntropy1.Program.scatter_read p.gradient
    (fun i : Fin p.block => CrossEntropy1.Source.gradientAddress p.toParameters row
      (CrossEntropy1.Source.column p.toParameters block i.val))
    (fun i => Cell.mk .real (backwardValues M p memory row block i))
    (fun i : Fin p.block => CrossEntropy1.Source.column p.toParameters block i.val<p.cols)
    unique (List.ofFn (fun i : Fin p.block => i)) memory lane active
  simpa only [List.mem_ofFn,exists_apply_eq_apply,if_true,Cell.read_same] using congrArg (fun c => c.read .real) h

theorem backward_frame {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) (region : RegionName) (offset : Nat)
    (outside : ∀ i : Fin p.block, CrossEntropy1.Source.column p.toParameters block i.val<p.cols →
      region≠p.gradient ∨ offset≠CrossEntropy1.Source.gradientAddress p.toParameters row
        (CrossEntropy1.Source.column p.toParameters block i.val)) :
    backward M p memory row block region offset=memory region offset := by
  apply CrossEntropy1.Program.scatter_frame
  intro i _ active
  exact outside i active

#axiomsClean scaled_integer_channel
#axiomsClean label_unchanged
#axiomsClean backward_ignored
#axiomsClean forward_frame
#axiomsClean backward_output
#axiomsClean backward_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Source
