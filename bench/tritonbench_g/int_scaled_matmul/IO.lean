import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.int_scaled_matmul.Proofs.RealSource
import VeriTile.Triton.Float.ProgramIO
import bench.tritonbench_g.int_scaled_matmul.Proofs.KernelModel

/-! Finite typed matrix channels for the two selected integer kernels.
The integer storage interpretation and accumulator wrap are retained. The scaled
kernel also retains its actual FP32 promotion, multiplication and total cast.
Read-only channels may alias; output separation remains a theorem premise. -/
namespace VeriTile.Bench.Repairs.IntScaledMatmul
open VeriTile.Triton FP.Structural Revision001
noncomputable section

namespace Formula
/-- Independent whole-K signed32 matrix product of signed8 operands. -/
def dot {M N K : Nat} (a : TileIndex [M,K] → Int)
    (b : TileIndex [K,N] → Int) (i : TileIndex [M,N]) : Int :=
  (BitVec.ofInt 32 (LogicalSum.sum (fun k => if hk : k<K then
    (BitVec.ofInt 8 (a (i.1,⟨k,hk⟩,PUnit.unit))).toInt *
      (BitVec.ofInt 8 (b (⟨k,hk⟩,i.2.1,PUnit.unit))).toInt else 0) K)).toInt

/-- The scalar suffix is defined on every FP32 word. Undefined casts retain
the original local conversion port, including NaN/infinity/out-of-range inputs. -/
def scaled (undefined : KernelModel.Word → BitVec 32) (acc : Int)
    (scale : KernelModel.Word) : Int :=
  FP.IntegerConversion.totalSigned (KernelModel.fallback undefined 32) 32
    (FP.mul Native.sourceConfig (FP.round .fp32 Native.sourceConfig (acc : Rat)) scale)
end Formula

namespace IntIO
abbrev inputs (p : BlockWord.Parameters) : List IOChannel :=
  [⟨.int,[p.rows,p.inner],p.input⟩,⟨.int,[p.inner,p.columns],p.weight⟩]
abbrev outputs (p : BlockWord.Parameters) : List IOChannel :=
  [⟨.int,[p.rows,p.columns],p.output⟩]

/-- The exact source and original one-axis launch; untouched local registers
and program-count metadata are inherited exactly as in the existing executor. -/
def invocation (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters)
    (pid : Nat) : Invocation KernelModel.Word where
  kernel := Kernels.unscaled p
  backend := BlockWord.backend rest
  pids := fun ax => if ax=0 then pid else 0

def io (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters) :
    ProgramIO KernelModel.Word KernelModel.Word where
  inputs := inputs p
  outputs := outputs p
  buffers := [p.input,p.weight,p.output]
  scratch := []
  read := fun i _ => by
    refine Fin.cases (fun j => j.1.val*p.inputRowStride+j.2.1.val*p.inputInnerStride) ?_ i
    intro k
    have hk : k=0 := Fin.eq_zero k
    subst k
    exact fun j => j.1.val*p.weightInnerStride+j.2.1.val*p.weightColumnStride
  inputMask := fun _ _ _ => 0<p.rows ∧ 0<p.columns
  write := fun o _ => by
    have ho : o=0 := Fin.eq_zero o
    subst o
    exact fun j => j.1.val*p.outputRowStride+j.2.1.val*p.outputColumnStride
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (List.range (Geometry.count (BlockWord.geometry p))).map (invocation rest p)
  encode := id
  observe := fun _ _ => id

def formula (p : BlockWord.Parameters) (vals : ChannelValues KernelModel.Word (inputs p)) :
    ChannelValues KernelModel.Word (outputs p) := by
  intro o
  have ho : o=0 := Fin.eq_zero o
  subst o
  exact Formula.dot (vals (0 : Fin 2)) (vals (1 : Fin 2))
end IntIO

namespace ScaledIO
abbrev inputs (p : ScaledWord.Parameters) : List IOChannel :=
  [⟨.int,[p.rows,p.inner],p.input⟩,⟨.int,[p.inner,p.columns],p.weight⟩,
    ⟨.real,[p.rows],p.scale⟩]
abbrev outputs (p : ScaledWord.Parameters) : List IOChannel :=
  [⟨.int,[p.rows,p.columns],p.output⟩]

def invocation (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (p : ScaledWord.Parameters) (pid : Nat) : Invocation KernelModel.Word where
  kernel := Kernels.scaled p
  backend := ScaledWord.backend rest Native.sourceConfig (KernelModel.fallback undefined)
  pids := fun ax => if ax=0 then pid else 0

/-- The source reads one FP32 scale per row; its retained scaleColumnStride
parameter is intentionally absent from the one-dimensional scale window. -/
def memory {α : Type} (p : ScaledWord.Parameters) : MemoryIO α where
  inputs := inputs p
  outputs := outputs p
  buffers := [p.input,p.weight,p.scale,p.output]
  scratch := []
  read := fun i _ => by
    refine Fin.cases (fun j => j.1.val*p.inputRowStride+j.2.1.val*p.inputInnerStride) ?_ i
    intro k
    refine Fin.cases (fun j => j.1.val*p.weightInnerStride+j.2.1.val*p.weightColumnStride) ?_ k
    intro k
    have hk : k=0 := Fin.eq_zero k
    subst k
    exact fun j => j.1.val*p.scaleRowStride
  inputMask := fun _ _ _ => 0<p.rows ∧ 0<p.columns
  write := fun o _ => by
    have ho : o=0 := Fin.eq_zero o
    subst o
    exact fun j => j.1.val*p.outputRowStride+j.2.1.val*p.outputColumnStride
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0

def io (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (p : ScaledWord.Parameters) : ProgramIO KernelModel.Word KernelModel.Word where
  toMemoryIO := memory p
  schedule := fun _ => (List.range (Geometry.count (ScaledWord.geometry p))).map
    (invocation rest undefined p)
  encode := id
  observe := fun _ _ => id

def formula (undefined : KernelModel.Word → BitVec 32) (p : ScaledWord.Parameters)
    (vals : ChannelValues KernelModel.Word (inputs p)) :
    ChannelValues KernelModel.Word (outputs p) := by
  intro o
  have ho : o=0 := Fin.eq_zero o
  subst o
  exact fun j => Formula.scaled undefined
    (Formula.dot (vals (0 : Fin 3)) (vals (1 : Fin 3)) j)
    (vals (2 : Fin 3) (j.1,PUnit.unit))
end ScaledIO
end
end VeriTile.Bench.Repairs.IntScaledMatmul

namespace VeriTile.Bench.Repairs.IntScaledMatmul.RealIO
open VeriTile.Triton FP.Structural Revision001
noncomputable section

abbrev inputs := ScaledIO.inputs
abbrev outputs := ScaledIO.outputs

/-- Independent ideal-Real formula. Integer operands and the accumulated dot retain
signed8/signed32 projections; scaling uses ordinary Real multiplication and the
Mathlib floor/ceiling definition of toward-zero conversion. -/
def resultValue (acc : Int) (scale : ℝ) : Int :=
  (BitVec.ofInt 32 (RealScaled.truncate ((acc : ℝ)*scale))).toInt

def invocation (rest : Algebra ℝ)
    (p : ScaledWord.Parameters) (pid : Nat) : Invocation ℝ where
  kernel := Kernels.scaled p
  backend := RealScaled.backend rest
  pids := fun ax => if ax=0 then pid else 0

/-- The source reads one FP32 scale per row; its retained scaleColumnStride
parameter is intentionally absent from the one-dimensional scale window. -/
def io (rest : Algebra ℝ)
    (p : ScaledWord.Parameters) : ProgramIO ℝ ℝ where
  toMemoryIO := ScaledIO.memory p
  schedule := fun _ => (List.range (Geometry.count (ScaledWord.geometry p))).map
    (invocation rest p)
  encode := id
  observe := fun _ _ => id

def formula (p : ScaledWord.Parameters) (vals : ChannelValues ℝ (inputs p)) :
    ChannelValues ℝ (outputs p) := by
  intro o
  have ho : o=0 := Fin.eq_zero o
  subst o
  exact fun j => resultValue (Formula.dot (vals (0 : Fin 3)) (vals (1 : Fin 3)) j)
    (vals (2 : Fin 3) (j.1,PUnit.unit))
end
end VeriTile.Bench.Repairs.IntScaledMatmul.RealIO

namespace VeriTile.Bench.Repairs.IntScaledMatmul.LogicalIO
open VeriTile.Triton FP.Structural Revision001
noncomputable section
structure IntegerInputs (p : BlockWord.Parameters) where
  a : Fin p.rows → Fin p.inner → Int
  b : Fin p.inner → Fin p.columns → Int
structure ScaledInputs (p : ScaledWord.Parameters) where
  a : Fin p.rows → Fin p.inner → Int
  b : Fin p.inner → Fin p.columns → Int
  rowScale : Fin p.rows → ℝ

def integerInputs (p : BlockWord.Parameters) : ChannelValues KernelModel.Word (IntIO.inputs p) ≃ IntegerInputs p where
  toFun xs := ⟨fun i k => xs 0 (i,k,()),fun k j => xs 1 (k,j,())⟩
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun j => xs.a j.1 j.2.1
    | ⟨1,_⟩ => fun j => xs.b j.1 j.2.1
  left_inv := by intro xs; funext o; fin_cases o <;> funext j <;> rcases j with ⟨r,c,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

def scaledInputs (p : ScaledWord.Parameters) : ChannelValues ℝ (RealIO.inputs p) ≃ ScaledInputs p where
  toFun xs := ⟨fun i k => xs 0 (i,k,()),fun k j => xs 1 (k,j,()),fun i => xs 2 (i,())⟩
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun j => xs.a j.1 j.2.1
    | ⟨1,_⟩ => fun j => xs.b j.1 j.2.1
    | ⟨2,_⟩ => fun j => xs.rowScale j.1
  left_inv := by
    intro xs; funext o; fin_cases o
    · funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
    · funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
    · funext j; rcases j with ⟨r,⟨⟩⟩; rfl
  right_inv := by intro xs; cases xs; rfl

def integerOutputs (p : BlockWord.Parameters) : ChannelValues KernelModel.Word (IntIO.outputs p) ≃ (Fin p.rows → Fin p.columns → Int) where
  toFun xs := fun i j => xs 0 (i,j,())
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun j => xs j.1 j.2.1
  left_inv := by intro xs; funext o; fin_cases o; funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
  right_inv := by intro xs; rfl

def scaledOutputs (p : ScaledWord.Parameters) : ChannelValues ℝ (RealIO.outputs p) ≃ (Fin p.rows → Fin p.columns → Int) where
  toFun xs := fun i j => xs 0 (i,j,())
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun j => xs j.1 j.2.1
  left_inv := by intro xs; funext o; fin_cases o; funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
  right_inv := by intro xs; rfl
end
end VeriTile.Bench.Repairs.IntScaledMatmul.LogicalIO

namespace VeriTile.Bench.Repairs.IntScaledMatmul.LogicalIO
open VeriTile.Triton FP.Structural
noncomputable section
open Revision001
def integerIO (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters) : LogicalProgramIO KernelModel.Word KernelModel.Word (IntegerInputs p) (Fin p.rows → Fin p.columns → Int) where
  core := IntIO.io rest p
  inputs := integerInputs p
  outputs := integerOutputs p

def scaledIO (rest : Algebra ℝ) (p : ScaledWord.Parameters) : LogicalProgramIO ℝ ℝ (ScaledInputs p) (Fin p.rows → Fin p.columns → Int) where
  core := RealIO.io rest p
  inputs := scaledInputs p
  outputs := scaledOutputs p
end
end VeriTile.Bench.Repairs.IntScaledMatmul.LogicalIO
