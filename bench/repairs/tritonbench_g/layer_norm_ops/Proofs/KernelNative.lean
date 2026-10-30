import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelExecution
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.LayerNormOps.KernelNative
open VeriTile.Triton FP.Structural
namespace Shared
export VeriTile.Bench.Repairs.FastLayernorm.KernelNative
  (Word Term Shape Reductions Instructions Schedule OutputSchedule algebra shape normalize eval literal32 withEpsilon withEpsilon_eq)
end Shared
open Shared
noncomputable section

/-- The admitted source uses FP32 storage, FP32 explicit arithmetic and FP32
inferred arithmetic. This type-directed lowering retains the source operator
tree and erases only identity FP32 casts. It inspects no input word. -/
def lower : Term → Term
  | .literal _ _ x => .literal (some .fp32) .real x
  | .binary _ _ op a b => .binary (some .fp32) .real op (lower a) (lower b)
  | .unary _ op a => .unary (some .fp32) op (lower a)
  | .cast _ _ _ a => lower a
  | .fromNat _ n => .fromNat (some .fp32) n
  | .fromInt _ n => .fromInt (some .fp32) n
  | .load a => lower a
  | t => t

/-- The original N argument is a runtime signed i32, converted to FP32 before
every mean/projection division. Range checks alone do not establish this law. -/
def countWord (I : Instructions) (columns : Nat) : Word :=
  I.fromInt (some .fp32) (Int.ofNat columns)
def CountConversion (I : Instructions) (columns : Nat) : Prop :=
  literal32 I (columns : ℝ)=countWord I columns

/-- Consume the runtime conversion in both source observation and independent
targets. Whole-record equality below also covers coinciding epsilon/unit
literals; no numerical transformation of an output or reduction is assumed. -/
def withCount (I : Instructions) (columns : Nat) : Instructions :=
  withEpsilon I (columns : ℝ) (countWord I columns)
theorem withCount_eq (I : Instructions) (columns : Nat)
    (conversion : CountConversion I columns) : withCount I columns=I :=
  withEpsilon_eq I (columns : ℝ) (countWord I columns) conversion

/-- A runtime N=1 remains a conversion followed by division in the selected
source. Do not borrow literal-denominator specialization from another task. -/
def runtimeSchedule (S : Schedule) : Schedule := { S with eraseUnitDivision := Bool.false }

def observe (I : Instructions) (S : Schedule) (columns : Nat) (cell : Cell Term) : Word :=
  eval (withCount I columns) (runtimeSchedule S)
    (normalize (runtimeSchedule S) (lower (cell.read .fp32)))

structure CountFacts (I : Instructions) (columns : Nat) : Prop where
  denominator : ∀ S,eval (withCount I columns) (runtimeSchedule S)
    (.literal (some .fp32) .real (columns : ℝ))=countWord I columns
  sourceObservation : ∀ S cell,observe I S columns cell=
    eval I (runtimeSchedule S) (normalize (runtimeSchedule S) (lower (cell.read .fp32)))
  division : ∀ S numerator,normalize (runtimeSchedule S)
    (.binary (some .fp32) .real .div numerator (.literal (some .fp32) .real (columns : ℝ)))=
    .binary (some .fp32) .real .div (normalize (runtimeSchedule S) numerator)
      (.literal (some .fp32) .real (columns : ℝ))

theorem count_facts (I : Instructions) (columns : Nat) (conversion : CountConversion I columns) :
    CountFacts I columns := by
  refine ⟨?_,?_,?_⟩
  · intro S
    rw [withCount_eq I columns conversion]
    exact (VeriTile.Bench.Repairs.FastLayernorm.KernelNative.eval_literal32 I _ _).trans conversion
  · intro S cell
    unfold observe
    rw [withCount_eq I columns conversion]
  · intro S numerator
    simp [VeriTile.Bench.Repairs.FastLayernorm.KernelNative.normalize,runtimeSchedule]

theorem count_one_division (S : Schedule) (numerator : Term) :
    normalize (runtimeSchedule S)
      (.binary (some .fp32) .real .div numerator (.literal (some .fp32) .real 1))=
    .binary (some .fp32) .real .div (normalize (runtimeSchedule S) numerator)
      (.literal (some .fp32) .real 1) := by
  simp [VeriTile.Bench.Repairs.FastLayernorm.KernelNative.normalize,runtimeSchedule]

def forwardY (R : Reductions) (I : Instructions) (S : Schedule)
    (p : ForwardParameters) (s : State Term) (i : TileIndex [p.block]) : Word :=
  observe I S p.columns (Forward.stored (algebra R) (KernelReference.forwardY (algebra R) p s i))
def rowDX (R : Reductions) (I : Instructions) (S : Schedule)
    (p : BackwardParameters) (s : State Term) (row : Nat) (i : TileIndex [p.block]) : Word :=
  observe I S p.columns (Forward.stored (algebra R) (KernelReference.rowDX (algebra R) p s row i))
def rowY (R : Reductions) (I : Instructions) (S : Schedule)
    (p : BackwardParameters) (s : State Term) (row : Nat) (i : TileIndex [p.block]) : Word :=
  observe I S p.columns (Forward.stored (algebra R) (KernelReference.rowY (algebra R) p s row i))
def partialDW (R : Reductions) (I : Instructions) (S : Schedule)
    (p : BackwardParameters) (s : State Term) (i : TileIndex [p.block]) : Word :=
  observe I S p.columns (Forward.stored (algebra R) ((KernelReference.partials (algebra R) p s).1 i))
def partialDB (R : Reductions) (I : Instructions) (S : Schedule)
    (p : BackwardParameters) (s : State Term) (i : TileIndex [p.block]) : Word :=
  observe I S p.columns (Forward.stored (algebra R) ((KernelReference.partials (algebra R) p s).2 i))

structure ForwardOutputs (R : Reductions) (I J : Instructions) (S : OutputSchedule)
    (p : ForwardParameters) (s t : State Term) : Prop where
  y : ∀ i : TileIndex [p.block],i.1.val<p.columns →
    observe I (S p.y (s.pids 0*p.yStride+i.1.val)) p.columns (t.mem p.y (s.pids 0*p.yStride+i.1.val))=
      forwardY R J (S p.y (s.pids 0*p.yStride+i.1.val)) p s i
  residual : p.storeResidual=Bool.true → ∀ i : TileIndex [p.block],i.1.val<p.columns →
    observe I (S p.residualOut (s.pids 0*p.residualOutStride+i.1.val)) p.columns
      (t.mem p.residualOut (s.pids 0*p.residualOutStride+i.1.val))=
      observe J (S p.residualOut (s.pids 0*p.residualOutStride+i.1.val)) p.columns
        (Forward.stored (algebra R) (Forward.z (algebra R) p s i))
  mean : p.rms=Bool.false → observe I (S p.mean (s.pids 0)) p.columns (t.mem p.mean (s.pids 0))=
    observe J (S p.mean (s.pids 0)) p.columns (Forward.stored (algebra R) (Forward.meanValue (algebra R) p s))
  rstd : observe I (S p.rstd (s.pids 0)) p.columns (t.mem p.rstd (s.pids 0))=
    observe J (S p.rstd (s.pids 0)) p.columns (Forward.stored (algebra R) (Forward.rstdValue (algebra R) p s))

structure BackwardOutputs (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (p : BackwardParameters) (s t : State Term) : Prop where
  dx : ∀ row,Backward.start p s≤row → row<Backward.stop p s →
    ∀ i : TileIndex [p.block],i.1.val<p.columns →
    observe I (S p.dx (row*p.dxStride+i.1.val)) p.columns (t.mem p.dx (row*p.dxStride+i.1.val))=
      rowDX R I (S p.dx (row*p.dxStride+i.1.val)) p s row i
  y : p.recompute=Bool.true → ∀ row,Backward.start p s≤row → row<Backward.stop p s →
    ∀ i : TileIndex [p.block],i.1.val<p.columns →
    observe I (S p.y (row*p.yStride+i.1.val)) p.columns (t.mem p.y (row*p.yStride+i.1.val))=
      rowY R I (S p.y (row*p.yStride+i.1.val)) p s row i
  residual : p.storeDres=Bool.true → ∀ row,Backward.start p s≤row → row<Backward.stop p s →
    ∀ i : TileIndex [p.block],i.1.val<p.columns →
    observe I (S p.dresIn (row*p.dresInStride+i.1.val)) p.columns (t.mem p.dresIn (row*p.dresInStride+i.1.val))=
      rowDX R I (S p.dresIn (row*p.dresInStride+i.1.val)) p s row i
  dw : ∀ i : TileIndex [p.block],i.1.val<p.columns →
    observe I (S p.dw (s.pids 0*p.columns+i.1.val)) p.columns (t.mem p.dw (s.pids 0*p.columns+i.1.val))=
      partialDW R I (S p.dw (s.pids 0*p.columns+i.1.val)) p s i
  db : p.hasBias=Bool.true → ∀ i : TileIndex [p.block],i.1.val<p.columns →
    observe I (S p.db (s.pids 0*p.columns+i.1.val)) p.columns (t.mem p.db (s.pids 0*p.columns+i.1.val))=
      partialDB R I (S p.db (s.pids 0*p.columns+i.1.val)) p s i

theorem forward (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (p : ForwardParameters) (s : State Term) (h : KernelExecution.ForwardAdmissible p)
    (count : CountConversion I p.columns)
    (epsilon : Word) (conversion : literal32 I p.epsilon=epsilon) :
    CountFacts I p.columns ∧ (∀ schedule,eval I schedule (.literal (some .fp32) .real p.epsilon)=epsilon) ∧
    ∃ t,KernelExecution.ForwardFacts (algebra R) p s t ∧
      ForwardOutputs R I (withEpsilon I p.epsilon epsilon) S p s t := by
  rw [withEpsilon_eq I p.epsilon epsilon conversion]
  obtain ⟨t,ht⟩ := KernelExecution.forward (algebra R) p s h
  refine ⟨count_facts I p.columns count,fun schedule =>
    (VeriTile.Bench.Repairs.FastLayernorm.KernelNative.eval_literal32 I schedule p.epsilon).trans conversion,
    t,ht,?_,?_,?_,?_⟩
  · intro i hi
    exact congrArg (observe I (S p.y (s.pids 0*p.yStride+i.1.val)) p.columns) (ht.y i hi)
  · intro hs i hi
    exact congrArg (observe I (S p.residualOut (s.pids 0*p.residualOutStride+i.1.val)) p.columns) (ht.residual hs i hi)
  · intro hr
    exact congrArg (observe I (S p.mean (s.pids 0)) p.columns) (ht.mean hr)
  · exact congrArg (observe I (S p.rstd (s.pids 0)) p.columns) ht.rstd

theorem backward (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (p : BackwardParameters) (s : State Term) (h : KernelReadback.Admissible p)
    (count : CountConversion I p.columns) :
    CountFacts I p.columns ∧ ∃ t,KernelExecution.BackwardFacts (algebra R) p s t ∧ BackwardOutputs R I S p s t := by
  obtain ⟨t,ht⟩ := KernelExecution.backward (algebra R) p s h
  refine ⟨count_facts I p.columns count,t,ht,?_,?_,?_,?_,?_⟩
  · intro row lo hr i hi
    exact congrArg (observe I (S p.dx (row*p.dxStride+i.1.val)) p.columns) (ht.dx row lo hr i hi)
  · intro hy row lo hr i hi
    exact congrArg (observe I (S p.y (row*p.yStride+i.1.val)) p.columns) (ht.y hy row lo hr i hi)
  · intro hd row lo hr i hi
    exact congrArg (observe I (S p.dresIn (row*p.dresInStride+i.1.val)) p.columns) (ht.residual hd row lo hr i hi)
  · intro i hi
    exact congrArg (observe I (S p.dw (s.pids 0*p.columns+i.1.val)) p.columns) (ht.dw i hi)
  · intro hb i hi
    exact congrArg (observe I (S p.db (s.pids 0*p.columns+i.1.val)) p.columns) (ht.db hb i hi)

#axiomsClean withCount_eq
#axiomsClean count_facts
#axiomsClean count_one_division
#axiomsClean forward
#axiomsClean backward
#specNonCircular forwardY avoiding [forwardKernel,backwardKernel,FP.Structural.exec,Forward.reference]
#specNonCircular rowDX avoiding [forwardKernel,backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
#specNonCircular rowY avoiding [forwardKernel,backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
#specNonCircular partialDW avoiding [forwardKernel,backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
#specNonCircular partialDB avoiding [forwardKernel,backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
end
end VeriTile.Bench.Repairs.LayerNormOps.KernelNative
