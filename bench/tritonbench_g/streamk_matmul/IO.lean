import bench.tritonbench_g.streamk_matmul.Proofs.Native
import bench.tritonbench_g.streamk_matmul.Proofs.ByteExecution
import VeriTile.Triton.Concurrent.EventProgramIO
import bench.tritonbench_g.streamk_matmul.SourceReads
import VeriTile.Triton.Float.LogicalProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

/-! Source-independent finite input arrays, output formulas and physical
windows. These are data for the pending shared event IO connection, not an
execution judgment or an alias around the old raw correctness proposition. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.LogicalIO
open Whole001 Source Checked
noncomputable section

/-- The original raw source admits arbitrary initial output and scratch.
Immutable A/B are logical arrays; their physical Tensor binding is separate. -/
structure Inputs (α : Type) (p : Parameters) where
  a : A p → α
  b : B p → α
  initialOutput : C p → α
  initialScratch : S p → α

def realFormula (p : Parameters) (values : Inputs ℝ p) : Returned.Cell p → ℝ :=
  fun cell=>Reference.matrixProduct p values.a values.b cell.1 cell.2

/-- Independent ordered K8 instruction formula. The full-tile branch retains
the original FP32→FP16→FP32 conversion, unlike the stream-tile branch. -/
def nativeFormula (reference : Native.Reference) (p : Parameters)
    (values : Inputs Native.Word p) : Returned.Cell p → Native.Word :=
  Returned.value p (Native.operations (Native.referenceInstructions reference)) values.a values.b

def outputAddress (p : Parameters) (layout : ByteExecution.Layout p)
    (cell : Returned.Cell p) : Nat := layout.outputBase+4*(Returned.index p cell).val

def scratchAddress (p : Parameters) (layout : ByteExecution.Layout p) (i : S p) : Nat :=
  layout.scratchBase+4*i.val

def scratchMask (p : Parameters) (layout : ByteExecution.Layout p) : Prop :=
  layout.scratchActive=Bool.true

def nativeByteFormula (reference : Native.Reference) (p : Parameters)
    (values : Inputs Native.Word p) (cell : Returned.Cell p) (byte : Fin 4) : BitVec 8 :=
  ByteMemory.byte (nativeFormula reference p values cell).bits byte.val

#specNonCircular realFormula avoiding [Source.call,Memory.run,Memory.step,Returned.tensor,Source.chunkValue]
#specNonCircular nativeFormula avoiding [Source.call,Memory.run,Memory.step,Returned.tensor]
#specNonCircular nativeByteFormula avoiding [Source.call,Memory.run,Memory.step,Returned.tensor]
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.LogicalIO

/-! Named logical IO of the complete decoded source. A/B retain their original
immutable-array interpretation. Initial C/S are arbitrary logical values; the
physical byte bindings are provided separately by the established transport.
Only active stream-K scratch is writable. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.EventIO
open Whole001 Source VeriTile.Triton Concurrent
noncomputable section

def arrayWindow (name : String) (region count : Nat) : Window :=
  {name:=name,region:=region,count:=count}

def aWindow (p : Parameters) := arrayWindow "A (FP32)" 0 (p.h.m*p.h.k)
def bWindow (p : Parameters) := arrayWindow "B (FP32)" 1 (p.h.k*p.h.n)
def cWindow (p : Parameters) := arrayWindow "C (FP32)" 0 (p.h.m*p.h.n)
def sInitialWindow (p : Parameters) := arrayWindow "initial partials (FP32)" 1 (p.h.programs*p.h.bm*p.h.bn)
def sWindow (p : Parameters) : Window :=
  {sInitialWindow p with name:="private partials (FP32)",active:=fun _=>decide (0<(Plan.select p.h).streamTiles)}

def scalarInputs (p : Parameters) : List ScalarInput :=
  [⟨.input,aWindow p⟩,⟨.input,bWindow p⟩,⟨.mem,cWindow p⟩,⟨.mem,sInitialWindow p⟩]

/-- The fixed shared interpreter receives the actual decoder's closed tape.
Only the scalar observation changes between Real and native interfaces. -/
def io (p : Parameters) (observe : V → O) : EventProgramIO Reader.Sources V O V where
  source := kernels
  decoded := (ClosedSource.sourceTape p).map Program.tape
  scalarInputs := scalarInputs p
  wordInputs := []
  scalarOutputs := [cWindow p]
  wordOutputs := []
  scalarScratch := [sWindow p]
  encode := id
  observe := fun _ _ => observe

abbrev Inputs (V : Type) (p : Parameters) := InputValues (scalarInputs p) [] V

def logicalInputs (p : Parameters) (values : Inputs V p) : LogicalIO.Inputs V p :=
  ⟨values.scalars ⟨0,by change 0<4; decide⟩,values.scalars ⟨1,by change 1<4; decide⟩,
    values.scalars ⟨2,by change 2<4; decide⟩,values.scalars ⟨3,by change 3<4; decide⟩⟩

/-- Row/column of each element of the complete M×N output, including the
vacuous empty-matrix case. -/
def cell (p : Parameters) (i : C p) : Returned.Cell p :=
  (⟨i.val/p.h.n,Nat.div_lt_of_lt_mul (by simpa only [Nat.mul_comm] using i.isLt)⟩,
    ⟨i.val%p.h.n,Nat.mod_lt _ (Nat.pos_of_ne_zero (by
      intro hn
      have hi:=i.isLt
      simp only [hn,Nat.mul_zero,Nat.not_lt_zero] at hi))⟩)

def outputIndex (p : Parameters) (port : Fin 1) (i : Fin ([cWindow p][port].count)) : C p := by
  have zero : port=0 := Fin.eq_zero port
  subst port
  exact i

def realArithmetic : Concurrent.Backend ℝ := ClosedSource.backend Arithmetic.real

def nativeArithmetic (reference : Native.Reference) : Concurrent.Backend Native.Word :=
  ClosedSource.backend (Native.operations (Native.referenceInstructions reference))

def realIO (p : Parameters) : EventProgramIO Reader.Sources ℝ ℝ ℝ := io p id

def realFormula (p : Parameters) (values : Inputs ℝ p) : OutputValues [cWindow p] [] ℝ where
  scalars := fun port i => LogicalIO.realFormula p (logicalInputs p values) (cell p (outputIndex p port i))
  words := fun port => Fin.elim0 port

/-- Both the native FP32 word and each of its four bytes are observed. This
introduces no floating rewrite, finite-only restriction or Real equality. -/
structure NativeObservation where
  word : Native.Word
  bytes : Fin 4 → BitVec 8

def observeNative (value : Native.Word) : NativeObservation :=
  ⟨value,fun j=>Checked.ByteMemory.byte value.bits j.val⟩

def nativeIO (p : Parameters) : EventProgramIO Reader.Sources Native.Word NativeObservation Native.Word :=
  io p observeNative

def nativeFormula (reference : Native.Reference) (p : Parameters)
    (values : Inputs Native.Word p) : OutputValues [cWindow p] [] NativeObservation where
  scalars := fun port i=>observeNative (LogicalIO.nativeFormula reference p (logicalInputs p values) (cell p (outputIndex p port i)))
  words := fun port => Fin.elim0 port

#specNonCircular realFormula avoiding [ClosedSource.sourceTape,Source.call,Memory.run,Memory.step]
#specNonCircular nativeFormula avoiding [ClosedSource.sourceTape,Source.call,Memory.run,Memory.step]
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.EventIO

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.MatrixIO
open Whole001 Source VeriTile.Triton Concurrent
noncomputable section

/-- Logical matrices and original arbitrary initial C/scratch snapshots. -/
structure Inputs (p : Parameters) where
  a : Fin p.h.m → Fin p.h.k → ℝ
  b : Fin p.h.k → Fin p.h.n → ℝ
  initialOutput : Fin p.h.m → Fin p.h.n → ℝ
  initialScratch : S p → ℝ

def inputs (p : Parameters) : (EventIO.realIO p).Inputs ≃ Inputs p where
  toFun := fun values => {
    a := fun r k => values.scalars (0 : Fin 4) (finProdFinEquiv (r,k))
    b := fun k c => values.scalars (1 : Fin 4) (finProdFinEquiv (k,c))
    initialOutput := fun r c => values.scalars (2 : Fin 4) (finProdFinEquiv (r,c))
    initialScratch := values.scalars (3 : Fin 4) }
  invFun := fun xs => {
    scalars := fun port => match port with
      | ⟨0,_⟩ => fun i => xs.a (finProdFinEquiv.symm i).1 (finProdFinEquiv.symm i).2
      | ⟨1,_⟩ => fun i => xs.b (finProdFinEquiv.symm i).1 (finProdFinEquiv.symm i).2
      | ⟨2,_⟩ => fun i => xs.initialOutput (finProdFinEquiv.symm i).1 (finProdFinEquiv.symm i).2
      | ⟨3,_⟩ => xs.initialScratch
      | ⟨n+4,h⟩ => False.elim (by simp [EventIO.realIO,EventIO.io,EventIO.scalarInputs] at h)
    words := fun port => Fin.elim0 port }
  left_inv := by
    intro values
    cases values with
    | mk scalars words =>
      apply congrArg₂ (fun s w => (⟨s,w⟩ : (EventIO.realIO p).Inputs))
      · funext port
        fin_cases port <;> funext i
        · exact congrArg (scalars (0 : Fin 4)) (finProdFinEquiv.apply_symm_apply i)
        · exact congrArg (scalars (1 : Fin 4)) (finProdFinEquiv.apply_symm_apply i)
        · exact congrArg (scalars (2 : Fin 4)) (finProdFinEquiv.apply_symm_apply i)
        · rfl
      · funext port
        exact Fin.elim0 port
  right_inv := by
    intro xs
    cases xs with
    | mk a b initialOutput initialScratch =>
      change Inputs.mk _ _ _ _ = Inputs.mk _ _ _ _
      congr 1 <;> funext r c <;> simp only [finProdFinEquiv.symm_apply_apply]

def outputs (p : Parameters) : (EventIO.realIO p).Outputs ≃
    (Fin p.h.m → Fin p.h.n → ℝ) where
  toFun := fun values r c => values.scalars (0 : Fin 1) (finProdFinEquiv (r,c))
  invFun := fun f => {
    scalars := fun port i => f (finProdFinEquiv.symm (EventIO.outputIndex p port i)).1
      (finProdFinEquiv.symm (EventIO.outputIndex p port i)).2
    words := fun port => Fin.elim0 port }
  left_inv := by
    intro values
    cases values with
    | mk scalars words =>
      apply congrArg₂ (fun s w => (⟨s,w⟩ : (EventIO.realIO p).Outputs))
      · funext port
        fin_cases port
        funext i
        exact congrArg (scalars (0 : Fin 1)) (finProdFinEquiv.apply_symm_apply i)
      · funext port
        exact Fin.elim0 port
  right_inv := by
    intro f
    funext r c
    change f (finProdFinEquiv.symm (finProdFinEquiv (r,c))).1
      (finProdFinEquiv.symm (finProdFinEquiv (r,c))).2 = f r c
    rw [finProdFinEquiv.symm_apply_apply]

def io (p : Parameters) : LogicalIO (EventIO.realIO p).Inputs (EventIO.realIO p).Outputs
    (Inputs p) (Fin p.h.m → Fin p.h.n → ℝ)
    ((EventIO.realIO p).Implements EventIO.realArithmetic) where
  inputs := inputs p
  outputs := outputs p

end
end VeriTile.Bench.Repairs.StreamK.TileChecked.MatrixIO
