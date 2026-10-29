import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelBackward
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.CrossEntropy1.KernelFloating
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

/-- A compiler reduction graph records each binary instruction and padding
leaf. Its validity is discrete input coverage, never a numerical equality. -/
inductive Tree (n : Nat) where
  | lane (i : Fin n)
  | pad
  | node (left right : Tree n)

def Tree.leaves : Tree n → List (Option (Fin n))
  | .lane i => [some i]
  | .pad => [none]
  | .node left right => left.leaves++right.leaves

def Tree.eval (combine : α → α → α) (padding : α) (xs : Fin n → α) : Tree n → α
  | .lane i => xs i
  | .pad => padding
  | .node left right => combine (left.eval combine padding xs) (right.eval combine padding xs)

structure Schedule (n : Nat) where
  tree : Tree n
  coverage : (tree.leaves.filterMap id).Perm (List.finRange n)

/-- Schedules may depend on the compiler's operation/layout metadata, not
numerical values. Sum and maximum schedules are kept separately. -/
abbrev Schedules := (precision : Option ComputeDType) → (shape : TileShape) →
  (ax : Fin shape.length) → (keepDims : Bool) →
  TileIndex (TileShape.reduceShape shape ax keepDims) → Schedule (TileShape.axisDim shape ax)

def reductionLanes {shape : TileShape} (ax : Fin shape.length) (keepDims : Bool)
    (xs : Values α .real shape) :
    TileIndex (TileShape.reduceShape shape ax keepDims) → Fin (TileShape.axisDim shape ax) → α :=
  match keepDims with
  | Bool.false => fun i k => xs (TileShape.insertAxisIndex shape ax i k)
  | Bool.true => fun i k => xs (TileShape.replaceAxisIndex shape ax i k)

/-- Native primitive/compiler boundary. Every literal, load, cast, exp/log,
and arithmetic instruction retains its selected FP32 call. Reduction calls
expand the chosen instruction trees explicitly. Exactness is for this operation
graph, with separate source add/multiply operations. Installed compiler lowering,
including cross-statement FMA contraction, is outside this theorem's claim. -/
def native (M : Algebra α) (sums maxima : Schedules) : Algebra α :=
  { M with
    reduceSum := fun precision {shape} ax keepDims xs i =>
      (sums precision shape ax keepDims i).tree.eval
        (M.binary precision .real .add) (M.literal precision .real 0)
        (reductionLanes ax keepDims xs i)
    reduceMax := fun precision {shape} ax keepDims xs i =>
      (maxima precision shape ax keepDims i).tree.eval
        (M.binary precision .real .max) M.negInf
        (reductionLanes ax keepDims xs i) }

/-- Independent native forward graph, on the actual initially bound words. -/
def forwardLSE (M : Algebra α) (sums maxima : Schedules) (p : Parameters) (initial : State α) : α :=
  KernelNative.lseValue (native M sums maxima) p initial

def forwardLoss (M : Algebra α) (sums maxima : Schedules) (p : Parameters) (initial : State α) : α :=
  KernelNative.lossValue (native M sums maxima) p initial

def backwardLane (M : Algebra α) (sums maxima : Schedules) (p : Parameters)
    (initial : State α) (lane : Fin p.block) : α :=
  KernelNative.gradientValue (native M sums maxima) p initial lane

/-- Successful typed FP32 forward execution has exactly the independently
ordered native graph's two output words, with the complete memory frame.
The tensor contract binds actual physical bytes, pointers and output ranks. -/
theorem forward (M : Algebra Word) (sums maxima : Schedules) (p : Parameters)
    (views : ForwardViews p) (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (initial : State Word) (inputs : ForwardInputs p views alloc bytes initial)
    (launch : Launch p) (pids : LaunchState p launch initial) :
    ∃ final, exec (KernelNative.engine (native M sums maxima)) (forwardSyntax p) initial=some final ∧
      fpValue final views.lse (launch.block*p.rows+launch.row)=forwardLSE M sums maxima p initial ∧
      fpValue final views.loss (launch.block*p.rows+launch.row)=forwardLoss M sums maxima p initial ∧
      (∀ region offset,
        (region≠p.loss ∨ offset≠p.lossBase+launch.block*p.rows+launch.row) →
        (region≠p.lse ∨ offset≠p.lseBase+launch.block*p.rows+launch.row) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,lseRead,lossRead,frame⟩ := KernelNative.forward_outputs
    (native M sums maxima) p initial inputs.parameters.geometry.widthPositive views.lseLogits views.distinctOutputs inputs.scalars inputs.scalarBinding
  simp only [pids.1,pids.2] at lseRead lossRead frame
  refine ⟨final,execution,?_,?_,frame⟩
  · simpa only [fpValue,output_index _ _ _ _ _ _ _ views.lseView launch.rowBound launch.blockBound,
      views.lseView.regionEq,forwardLSE] using lseRead
  · simpa only [fpValue,output_index _ _ _ _ _ _ _ views.lossView launch.rowBound launch.blockBound,
      views.lossView.regionEq,forwardLoss] using lossRead

/-- Native backward output words, including literal-zero ignored rows,
masked tails, and the permitted in-place output allocation. -/
theorem backward (M : Algebra Word) (sums maxima : Schedules) (p : Parameters)
    (views : BackwardViews p) (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (initial : State Word) (inputs : BackwardInputs p views alloc bytes initial)
    (launch : Launch p) (pids : LaunchState p launch initial) :
    ∃ final, exec (KernelNative.engine (native M sums maxima)) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
        fpValue final views.gradient (launch.row*p.cols+(launch.block*p.block+lane.val))=
          backwardLane M sums maxima p initial lane) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
          region≠p.gradient ∨ offset≠p.gradientBase+launch.row*p.gradientStride+(launch.block*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,reads,frame⟩ := KernelNative.backward_outputs (native M sums maxima) p initial inputs.scalars inputs.scalarBinding
  simp only [KernelNative.gradientOffset,pids.1,pids.2] at reads frame
  refine ⟨final,execution,?_,frame⟩
  intro lane active
  simpa only [fpValue,matrix_index _ _ _ _ _ _ _ _ views.gradientView launch.rowBound active,
    views.gradientView.regionEq,backwardLane] using reads lane active

#axiomsClean forward
#axiomsClean backward
#specNonCircular forwardLSE avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#specNonCircular forwardLoss avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#specNonCircular backwardLane avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.KernelFloating
