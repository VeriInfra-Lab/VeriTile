import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.IntegerBinding
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Candidate
import VeriTile.Triton.Float.IntegerProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.IO
open VeriTile.Triton FP.Structural
noncomputable section

abbrev inputs (p : Launch) : List IOChannel := [⟨.real,[p.rows,p.columns],"x"⟩]
abbrev outputs (p : Launch) : List IOChannel :=
  [⟨.int,[p.rows,p.columns],"q"⟩,⟨.fp16,[p.rows],"m"⟩]

def invocation (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (row : Nat) : IntegerInvocation α where
  kernel := source "x" "q" "m" p.columns p.width
  backend := KernelExecution.backend M equal
  pids := fun _ => row
  setPids := fun ax => decide (ax=0)
  integerStorage := fun _ => none
  runtimeRegisters := QuantizeIntegerStage.outputRegister
  round := some round

/-- Complete arbitrary row permutation. FP32 loads use the accepted erased
Real cells; signed64 result cells are observed as INT8 codes by byteView.
The FP16 maximum retains its own cell tag and source conversion. -/
abbrev quantizerIO (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (schedule : Schedule p) : IntegerProgramIO α α where
  inputs := inputs p
  outputs := outputs p
  scratch := []
  buffers := ["x","q","m"]
  read := fun i _ => match i with
    | ⟨0,_⟩ => fun j => j.1.val*p.columns+j.2.1.val
    | ⟨_+1,h⟩ => False.elim (by simp at h)
  inputMask := fun _ _ _ => True
  write := fun o _ => match o with
    | ⟨0,_⟩ => fun j => j.1.val*p.columns+j.2.1.val
    | ⟨1,_⟩ => fun j => j.1.val
    | ⟨_+2,h⟩ => False.elim (by simp at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => schedule.order.map (invocation M round equal p)
  encode := id
  observe := fun _ _ => id

def byteView (p : Launch) (o : Fin (outputs p).length) (_ : TileIndex (outputs p)[o].shape) :
    IntegerOutputView := if o.val=0 then .unsignedWidth 8 else .identity

def values [Inhabited α] (p : Launch) (vals : ChannelValues α (inputs p)) (r c : Nat) : α :=
  if hr : r<p.rows then if hc : c<p.columns then vals 0 (⟨r,hr⟩,⟨c,hc⟩,()) else default else default

/-- Independent quantized codes and row maxima supplied by the reader. -/
def result [Inhabited α] (p : Launch) (quantized : (Nat → Nat → α) → Nat → Nat → Int)
    (maximum : (Nat → Nat → α) → Nat → α) (vals : ChannelValues α (inputs p)) :
    ChannelValues α (outputs p) := fun o => match o with
    | ⟨0,_⟩ => fun j => quantized (values p vals) j.1.val j.2.1.val
    | ⟨1,_⟩ => fun j => maximum (values p vals) j.1.val
    | ⟨_+2,h⟩ => False.elim (by simp at h)


/-- Both genuinely different output channels remain explicit and typed. -/
structure LogicalOutputs (α : Type) (p : Launch) where
  quantized : Fin p.rows → Fin p.columns → Int
  maximum : Fin p.rows → α

def logicalInputs {α : Type} (p : Launch) :
    ChannelValues α (inputs p) ≃ (Fin p.rows → Fin p.columns → α) :=
  (ChannelValues.singleEquiv α _).trans (Values.matrixEquiv α .real _ _)

def logicalOutputs {α : Type} (p : Launch) :
    ChannelValues α (outputs p) ≃ LogicalOutputs α p where
  toFun := fun ys => {
    quantized := fun r c => ys (0 : Fin 2) (r,c,())
    maximum := fun r => ys (1 : Fin 2) (r,()) }
  invFun := fun ys => Fin.cases (fun j => ys.quantized j.1 j.2.1)
    (Fin.cases (fun j => ys.maximum j.1) (fun i => i.elim0))
  left_inv := by intro ys; funext port j; fin_cases port <;> rfl
  right_inv := by rintro ⟨q,m⟩; rfl

/-- The canonical total logical row extension; no physical address occurs. -/
def extendRow {α : Type} [Inhabited α] {N : Nat} (xs : Fin N → α) (col : Nat) : α :=
  if h : col<N then xs ⟨col,h⟩ else default

/-- Retains llrint64/INT8 observation and the FP16 maximum channel of the exact
previous IntegerProgramIO judgment, including the arbitrary complete row order. -/
def logicalIO [Inhabited α] (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (schedule : Schedule p) :
    LogicalIO (ChannelValues α (inputs p)) (ChannelValues α (outputs p))
      (Fin p.rows → Fin p.columns → α) (LogicalOutputs α p)
      ((quantizerIO M round equal p schedule).ObservesInteger (byteView p)) where
  inputs := logicalInputs p
  outputs := logicalOutputs p

end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.IO

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.StoreSchedule
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

def invocation (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (row : Nat) : IntegerInvocation α :=
  { IO.invocation M round equal p row with
    kernel := Candidate.source "x" "q" "m" p.columns p.width }

def variant (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (schedule : Schedule p) :
    IntegerProgramIO.Variant (IO.quantizerIO M round equal p schedule) where
  schedule := fun _ => schedule.order.map (invocation M round equal p)
  observe := fun _ _ => id
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.StoreSchedule
