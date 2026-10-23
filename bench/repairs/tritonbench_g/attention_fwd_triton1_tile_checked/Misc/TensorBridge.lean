import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Correct
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.InputBinding
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.PhysicalNative
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.NumericalCompatibility

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.TensorBridge
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

def realScale (c : Shape) : ℝ := 1/Real.sqrt (c.D:ℝ)
theorem realScale_ne_zero (c : Shape) : realScale c≠0 := by
  have h : (0:ℝ)<c.D := Nat.cast_pos.mpr c.dim_pos
  exact div_ne_zero one_ne_zero (ne_of_gt (Real.sqrt_pos.2 h))

/-- Independent Real target using the actual input Tensor storage reads. -/
def realOutput (c : Shape) (t : Tensors) (before : Cells ℝ) (head row j : Nat) : ℝ :=
  MathTarget.output c (actualInputs c t before) (realScale c) head row j
def realHistory (c : Shape) (t : Tensors) (before : Cells ℝ) (head chunk i j : Nat) : ℝ :=
  MathTarget.history c (actualInputs c t before) head chunk i j

structure CellFrame {A} (c : Shape) (abi : ABI c) (storeFlag : Bool)
    (before after : Cells A) : Prop where
  outside : ∀ address, ¬Physical.WordAt abi.o address →
    (storeFlag=Bool.false ∨ ¬Physical.WordAt abi.h address) → after address=before address
  inputs : ∀ a∈[abi.q,abi.k,abi.v], ∀ address, a.Contains (address/4) → after address=before address
  disabledHistory : storeFlag=Bool.false → ∀ address,
    abi.h.Contains (address/4) → after address=before address

structure RealSpecification (c : Shape) (abi : ABI c) (t : Tensors) (storeFlag : Bool)
    (before after : Cells ℝ) : Prop where
  output : ∀ head row j, head<c.heads → row<c.T → j<c.D →
    logical after t.output (c.mainOffset head row j)=realOutput c t before head row j
  history : storeFlag=Bool.true → ∀ head chunk i j,
    head<c.heads → chunk<c.chunks → i<c.D → j<c.D →
    logical after t.history (c.historyOffset head chunk i j)=realHistory c t before head chunk i j
  frame : CellFrame c abi storeFlag before after

namespace Correct

theorem frame {A} [Inhabited A] (ops : Ops A) (literal : ℝ → A)
    (c : Shape) (abi : ABI c) (scale : ℝ) (storeFlag ifcond : Bool)
    (before : Cells A) (rest : State (Term A)) :
    CellFrame c abi storeFlag before (Physical.result ops literal c abi scale storeFlag ifcond before rest) := by
  refine ⟨Physical.frame ops literal c abi scale storeFlag ifcond before rest,
    TensorEffect.input_frame ops literal c abi scale storeFlag ifcond before rest,?_⟩
  intro hs
  subst storeFlag
  exact TensorEffect.history_disabled ops literal c abi scale ifcond before rest

theorem output (c : Shape) (abi : ABI c) (t : Tensors) (b : TensorBinding c abi t)
    (storeFlag ifcond : Bool) (admit : InputContract c (t.arguments storeFlag ifcond) b.pointers)
    (before : Cells ℝ) (rest : State (Term ℝ)) (head row j : Nat)
    (hh : head<c.heads) (hr : row<c.T) (hj : j<c.D) :
    logical (Physical.result realOps id c abi (realScale c) storeFlag ifcond before rest) t.output
      (c.mainOffset head row j)=realOutput c t before head row j := by
  rw [TensorEffect.output realOps id rfl c abi t b (realScale c) storeFlag ifcond admit before rest head row ⟨j,hj⟩ hh hr]
  change Reference.output realOps c (tensorInputs c abi before) (realScale c) head (row/32) ifcond (row%32) j=_
  rw [InputBinding.output realOps c _ (actualInputs c t before) (realScale c) head (row/32) ifcond
    (InputBinding.actual c abi t b storeFlag ifcond admit before head hh)]
  exact RealProof.output_target c _ (realScale c) head row j ifcond hr hj

theorem history (c : Shape) (abi : ABI c) (t : Tensors) (b : TensorBinding c abi t)
    (ifcond : Bool) (admit : InputContract c (t.arguments Bool.true ifcond) b.pointers)
    (before : Cells ℝ) (rest : State (Term ℝ)) (head chunk i j : Nat)
    (hh : head<c.heads) (hc : chunk<c.chunks) (hi : i<c.D) (hj : j<c.D) :
    logical (Physical.result realOps id c abi (realScale c) Bool.true ifcond before rest) t.history
      (c.historyOffset head chunk i j)=realHistory c t before head chunk i j := by
  rw [TensorEffect.history realOps id rfl c abi t b (realScale c) ifcond admit before rest head chunk ⟨i,hi⟩ ⟨j,hj⟩ hh hc]
  change Reference.carry realOps c (tensorInputs c abi before) (realScale c) head ifcond chunk i j=_
  rw [InputBinding.carry realOps c _ (actualInputs c t before) (realScale c) head ifcond chunk
    (InputBinding.actual c abi t b Bool.true ifcond admit before head hh)]
  exact RealProof.carry_target c _ (realScale c) head ifcond chunk i j hi hj

/-- Kernel-only end-to-end Real theorem. Shared execution, Tensor reads,
frame, checked machine arithmetic and tile admission are conclusions. -/
theorem kernel_correct (c : Shape) (abi : ABI c) (t : Tensors) (b : TensorBinding c abi t)
    (storeFlag ifcond : Bool) (admit : InputContract c (t.arguments storeFlag ifcond) b.pointers)
    (before : Cells ℝ) (rest : State (Term ℝ)) :
    Physical.run realOps id c abi (realScale c) storeFlag ifcond before rest=
      some (Physical.result realOps id c abi (realScale c) storeFlag ifcond before rest) ∧
    RealSpecification c abi t storeFlag before
      (Physical.result realOps id c abi (realScale c) storeFlag ifcond before rest) ∧
    Machine.Closed c abi ∧ (¬Host.Empty c → c.D*c.D≤1048576) := by
  refine ⟨Physical.success _ _ _ _ _ _ _ _ _,⟨?_,?_,frame _ _ _ _ _ _ _ _ _⟩,
    machine_closed c _ abi b.pointers admit,static_tile_bound c _ b.pointers admit⟩
  · exact output c abi t b storeFlag ifcond admit before rest
  · intro hs
    subst storeFlag
    exact history c abi t b ifcond admit before rest

#axiomsClean kernel_correct
#specNonCircular realOutput avoiding [FP.Structural.exec,Physical.run,Physical.result,Reference.output]
#specNonCircular realHistory avoiding [FP.Structural.exec,Physical.run,Physical.result,Reference.carry]
#specNonCircular TensorBinding avoiding [Physical.run,Physical.result,Host.AllocationReceipt]
#guard_msgs (drop info) in
#auditModuleAxioms
end Correct
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.TensorBridge

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.TensorBridge
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

/-- The actual Torch FP32 logical read, including its view-flag semantics. -/
def tensorRead (cells : Cells Word) (p : Payload) (i : Nat) : Word :=
  (VeriTile.Torch.FP32.logicalValue (fun address => ⟨(cells address).read .fp32⟩)
    p.tensor p.laws i).bits

theorem logical_tensorRead (cells : Cells Word) (p : Payload)
    (ordinary : p.tensor.metadata.isNeg=Bool.false) (i : Nat) :
    logical cells p i=tensorRead cells p i := by
  simp [logical,tensorRead,VeriTile.Torch.FP32.logicalValue,ordinary]

def fpInputs (c : Shape) (t : Tensors) (before : Cells Word) : Inputs Word :=
  ⟨fun h r d => tensorRead before t.q (c.mainOffset h r d),
   fun h r d => tensorRead before t.k (c.mainOffset h r d),
   fun h r d => tensorRead before t.v (c.mainOffset h r d)⟩

theorem actualInputs_fp (c : Shape) (abi : ABI c) (t : Tensors) (b : TensorBinding c abi t)
    (storeFlag ifcond : Bool) (admit : InputContract c (t.arguments storeFlag ifcond) b.pointers)
    (before : Cells Word) : actualInputs c t before=fpInputs c t before := by
  have q := input_facts c abi t b storeFlag ifcond admit t.q (by simp)
  have k := input_facts c abi t b storeFlag ifcond admit t.k (by simp)
  have v := input_facts c abi t b storeFlag ifcond admit t.v (by simp)
  simp only [actualInputs,fpInputs,logical_tensorRead _ t.q q.ordinary,
    logical_tensorRead _ t.k k.ordinary,logical_tensorRead _ t.v v.ordinary]

namespace Native
/-- Native binding of the quote's two scalar literal sites. Zero has the
FP32 +0 word; the nonzero scale slot uses the actual host/JIT scale word.
No equality between this word and an ideal real scale is asserted. -/
def literal (actualScale : Word) (x : ℝ) : Word := if x=0 then 0 else actualScale
@[simp] theorem literal_zero (actualScale : Word) : literal actualScale 0=0 := by simp [literal]
@[simp] theorem literal_scale (c : Shape) (actualScale : Word) :
    literal actualScale (realScale c)=actualScale := by simp [literal,realScale_ne_zero]

def result (native : ISA) (c : Shape) (abi : ABI c) (actualScale : Word)
    (storeFlag ifcond : Bool) (before : Cells Word) (rest : State (Term Word)) : Cells Word :=
  Physical.result (wordOps native) (literal actualScale) c abi (realScale c) storeFlag ifcond before rest

def output (c : Shape) (t : Tensors) (before : Cells Word) (actualScale : Word)
    (head row j : Nat) (ifcond : Bool) : Word :=
  SourceFP.output c (fpInputs c t before) actualScale head row j ifcond
def history (c : Shape) (t : Tensors) (before : Cells Word) (actualScale : Word)
    (head chunk i j : Nat) (ifcond : Bool) : Word :=
  SourceFP.history c (fpInputs c t before) actualScale head chunk i j ifcond
end Native

structure BitSpecification (c : Shape) (abi : ABI c) (t : Tensors) (actualScale : Word)
    (storeFlag ifcond : Bool) (before after : Cells Word) : Prop where
  output : ∀ head row j, head<c.heads → row<c.T → j<c.D →
    tensorRead after t.output (c.mainOffset head row j)=Native.output c t before actualScale head row j ifcond
  history : storeFlag=Bool.true → ∀ head chunk i j,
    head<c.heads → chunk<c.chunks → i<c.D → j<c.D →
    tensorRead after t.history (c.historyOffset head chunk i j)=Native.history c t before actualScale head chunk i j ifcond
  frame : CellFrame c abi storeFlag before after

namespace FPEquiv

theorem output (native : ISA) (dom : NumericDomain) (c : Shape) (abi : ABI c)
    (t : Tensors) (b : TensorBinding c abi t) (actualScale : Word) (storeFlag ifcond : Bool)
    (admit : InputContract c (t.arguments storeFlag ifcond) b.pointers)
    (before : Cells Word) (rest : State (Term Word))
    (numeric : BitNumerical native dom c (fpInputs c t before) actualScale ifcond)
    (head row j : Nat) (hh : head<c.heads) (hr : row<c.T) (hj : j<c.D) :
    tensorRead (Native.result native c abi actualScale storeFlag ifcond before rest) t.output
      (c.mainOffset head row j)=Native.output c t before actualScale head row j ifcond := by
  rw [←logical_tensorRead _ _ b.oOrdinary]
  unfold Native.result
  rw [TensorEffect.output (wordOps native) (Native.literal actualScale) (by simp [wordOps])
    c abi t b (realScale c) storeFlag ifcond admit before rest head row ⟨j,hj⟩ hh hr]
  change Reference.output (wordOps native) c (tensorInputs c abi before)
    (Native.literal actualScale (realScale c)) head (row/32) ifcond (row%32) j=_
  rw [Native.literal_scale,InputBinding.output (wordOps native) c _ (actualInputs c t before) actualScale
    head (row/32) ifcond (InputBinding.actual c abi t b storeFlag ifcond admit before head hh),
    actualInputs_fp c abi t b storeFlag ifcond admit before]
  rw [←AttentionFwdTriton1.Whole001.Graph.eval_output native c (fpInputs c t before) actualScale
    head (row/32) ifcond (row%32) j]
  rw [SourceFP.transport native dom numeric.scalarLaws _
    (numeric.admitted.output head (row/32) (row%32) j hh (c.chunk_of_row row hr)
      (Nat.mod_lt row (by decide)) hj),AttentionFwdTriton1.Whole001.Graph.eval_output]
  rfl

theorem history (native : ISA) (dom : NumericDomain) (c : Shape) (abi : ABI c)
    (t : Tensors) (b : TensorBinding c abi t) (actualScale : Word) (ifcond : Bool)
    (admit : InputContract c (t.arguments Bool.true ifcond) b.pointers)
    (before : Cells Word) (rest : State (Term Word))
    (numeric : BitNumerical native dom c (fpInputs c t before) actualScale ifcond)
    (head chunk i j : Nat) (hh : head<c.heads) (hc : chunk<c.chunks) (hi : i<c.D) (hj : j<c.D) :
    tensorRead (Native.result native c abi actualScale Bool.true ifcond before rest) t.history
      (c.historyOffset head chunk i j)=Native.history c t before actualScale head chunk i j ifcond := by
  rw [←logical_tensorRead _ _ b.hOrdinary]
  unfold Native.result
  rw [TensorEffect.history (wordOps native) (Native.literal actualScale) (by simp [wordOps])
    c abi t b (realScale c) ifcond admit before rest head chunk ⟨i,hi⟩ ⟨j,hj⟩ hh hc]
  change Reference.carry (wordOps native) c (tensorInputs c abi before)
    (Native.literal actualScale (realScale c)) head ifcond chunk i j=_
  rw [Native.literal_scale,InputBinding.carry (wordOps native) c _ (actualInputs c t before) actualScale
    head ifcond chunk (InputBinding.actual c abi t b Bool.true ifcond admit before head hh),
    actualInputs_fp c abi t b Bool.true ifcond admit before]
  rw [←AttentionFwdTriton1.Whole001.Graph.eval_carry native c (fpInputs c t before) actualScale
    head ifcond chunk i j]
  rw [SourceFP.transport native dom numeric.scalarLaws _
    (numeric.admitted.carry head chunk i j hh (Nat.le_of_lt hc) hi hj),
    AttentionFwdTriton1.Whole001.Graph.eval_carry]
  rfl

/-- Actual Tensor FP32 kernel endpoint. Numerical assumptions contain only
local mulRN/fmaRN conformance and reached operand domains. The shared quote
execution, complete result and all-dtype caller frame are proved here. -/
theorem kernel_equiv (native : ISA) (dom : NumericDomain) (c : Shape) (abi : ABI c)
    (t : Tensors) (b : TensorBinding c abi t) (actualScale : Word) (storeFlag ifcond : Bool)
    (admit : InputContract c (t.arguments storeFlag ifcond) b.pointers)
    (before : Cells Word) (rest : State (Term Word))
    (numeric : BitNumerical native dom c (fpInputs c t before) actualScale ifcond) :
    Physical.run (wordOps native) (Native.literal actualScale) c abi (realScale c)
      storeFlag ifcond before rest=some (Native.result native c abi actualScale storeFlag ifcond before rest) ∧
    BitSpecification c abi t actualScale storeFlag ifcond before
      (Native.result native c abi actualScale storeFlag ifcond before rest) ∧
    Machine.Closed c abi ∧ (¬Host.Empty c → c.D*c.D≤1048576) := by
  refine ⟨Physical.success _ _ _ _ _ _ _ _ _,⟨?_,?_,Correct.frame _ _ _ _ _ _ _ _ _⟩,
    machine_closed c _ abi b.pointers admit,static_tile_bound c _ b.pointers admit⟩
  · exact output native dom c abi t b actualScale storeFlag ifcond admit before rest numeric
  · intro hs
    subst storeFlag
    exact history native dom c abi t b actualScale ifcond admit before rest numeric

/-- Closed witness: the concrete software FP32 profile satisfies the same
endpoint for all admitted shapes, input words and actual scale words. -/
theorem source_fp32 (c : Shape) (abi : ABI c) (t : Tensors) (b : TensorBinding c abi t)
    (actualScale : Word) (storeFlag ifcond : Bool)
    (admit : InputContract c (t.arguments storeFlag ifcond) b.pointers)
    (before : Cells Word) (rest : State (Term Word)) :
    BitSpecification c abi t actualScale storeFlag ifcond before
      (Native.result SourceFP.scalar c abi actualScale storeFlag ifcond before rest) :=
  (kernel_equiv SourceFP.scalar SourceFP.allOperands c abi t b actualScale storeFlag ifcond admit before rest
    (reference_witness c (fpInputs c t before) actualScale ifcond)).2.1

#axiomsClean kernel_equiv
#axiomsClean source_fp32
#specNonCircular Native.output avoiding [FP.Structural.exec,Physical.run,Physical.result,Native.result]
#specNonCircular Native.history avoiding [FP.Structural.exec,Physical.run,Physical.result,Native.result]
#specNonCircular BitNumerical avoiding [FP.Structural.exec,Physical.run,Physical.result,Native.result,MathTarget.output]
#guard_msgs (drop info) in
#auditModuleAxioms
end FPEquiv
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.TensorBridge

