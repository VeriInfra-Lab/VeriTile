import bench.tritonbench_g.quant_transpose_kernel.Proofs.KernelModels
import VeriTile.Triton.Float.IntegerOutputIO
import VeriTile.Triton.Float.IntegerProgramEquivalenceSyntax
import VeriTile.Triton.Memory.LogicalIO
import Mathlib.Tactic.FinCases

/-! The complete grouped transpose has an arbitrary strided FP32 matrix input,
one FP32 inverse-scale cell, and a dense transposed INT8 output. The existing
integer interpreter keeps signed register cells for computed-pointer stores;
the public byte view is specified separately from those raw cells. -/
namespace VeriTile.Bench.Repairs.QuantTransposeKernel.IO
open VeriTile.Triton FP.Structural KernelSource KernelModels
noncomputable section

def launch (algebra : Algebra α) (round : α → BitVec 64) (p : Parameters)
    (version : Bool) (pid : Nat) : IntegerInvocation α where
  kernel := kernel version p
  backend := algebra
  pids := fun _ => pid
  setPids := fun ax => decide (ax=0)
  integerStorage := QuantizeIntegerStage.byteOutput ⟨p.output⟩
  runtimeRegisters := QuantizeIntegerStage.outputRegister
  round := some round

def schedule (algebra : Algebra α) (round : α → BitVec 64) (p : Parameters)
    (version : Bool) : List (IntegerInvocation α) :=
  (List.range (QuantTransposeFinite.gridM p.M * QuantTransposeFinite.gridN p.N)).map
    (launch algebra round p version)

def grid (algebra : Algebra α) (round : α → BitVec 64) (p : Parameters)
    (version : Bool) : IntegerProgramIO α α where
  inputs := [⟨.real,[p.M,p.N],⟨p.x⟩⟩,⟨.real,[],⟨p.scale⟩⟩]
  outputs := [⟨.int,[p.M,p.N],⟨p.output⟩⟩]
  scratch := []
  buffers := [⟨p.x⟩,⟨p.scale⟩,⟨p.output⟩]
  read := fun i => Fin.cases (fun _ j => j.1.val*p.sam+j.2.1.val*p.san)
    (Fin.cases (fun _ _ => 0) (fun i => Fin.elim0 i)) i
  inputMask := fun _ _ _ => True
  write := fun o => Fin.cases (fun _ j => j.2.1.val*p.M+j.1.val) (fun i => Fin.elim0 i) o
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => schedule algebra round p version
  encode := id
  observe := fun _ _ => id

def commuted (algebra : Algebra α) (round : α → BitVec 64) (p : Parameters) :
    IntegerProgramIO.Variant (grid algebra round p Bool.false) where
  schedule := fun _ => schedule algebra round p Bool.true
  observe := fun _ _ => id

def real (rest : Algebra ℝ) (p : Parameters) (version : Bool) : IntegerProgramIO ℝ ℝ :=
  grid (realAlgebra rest) realRound p version

def native (multiply : Word → Word → Word) (rest : Algebra Word) (p : Parameters)
    (version : Bool) : IntegerProgramIO Word Word :=
  grid (nativeAlgebra multiply rest) nativeRound p version

/-- Fixed byte-bit observation of the actual INT8 output. The raw integer
interpreter cell and its exact frame are preserved by the shared judgment. -/
def byteView (io : IntegerProgramIO α β) :
    (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView :=
  fun _ _ => .unsignedWidth 8

/-- Actual complete source schedule with the fixed INT8 byte observation.
The full public layout and unrestricted logical input domain are type data. -/
def originalProgram (multiply : Word → Word → Word) (rest : Algebra Word)
    (p : Parameters) :=
  IntegerProgramIO.ObservedImplementation.original (native multiply rest p Bool.false)
    (fun _ => True) (byteView (native multiply rest p Bool.false))

/-- The existing integer-index candidate, with the same complete grid and
byte observer. No expected result or custom executor is stored in this view. -/
def commutedProgram (multiply : Word → Word → Word) (rest : Algebra Word)
    (p : Parameters) :=
  IntegerProgramIO.ObservedImplementation.fromVariant (fun _ => True)
    (byteView (native multiply rest p Bool.false))
    (commuted (nativeAlgebra multiply rest) nativeRound p)

/-- Independent scalar reference lifted to every logical matrix coordinate. -/
def formula (algebra : Algebra α) (round : α → BitVec 64) (p : Parameters)
    (version : Bool) (f : α → α → Int) :
    (grid algebra round p version).Inputs → (grid algebra round p version).Outputs :=
  fun vals o => Fin.cases
    (fun j => f (vals (0 : Fin 2) j) (vals (1 : Fin 2) PUnit.unit))
    (fun i => Fin.elim0 i) o

end
end VeriTile.Bench.Repairs.QuantTransposeKernel.IO

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.LogicalView
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
structure Inputs (p : Parameters) where
  values : Fin p.M → Fin p.N → ℝ
  inverse : ℝ

def inputEquiv (rest : Algebra ℝ) (p : Parameters) (version : Bool) :
    (IO.real rest p version).Inputs ≃ Inputs p where
  toFun := fun xs => ⟨fun r c => xs ⟨0,by simp [IO.real,IO.grid]⟩ (r,c,()),xs ⟨1,by simp [IO.real,IO.grid]⟩ ()⟩
  invFun := fun xs => Fin.cases (fun j => xs.values j.1 j.2.1)
    (Fin.cases (fun _ => xs.inverse) (fun k => k.elim0))
  left_inv := by intro xs; funext o j; fin_cases o <;> rfl
  right_inv := by intro xs; cases xs; rfl

/-- Output coordinates are [column,row], matching the actual transpose. -/
def outputEquiv (rest : Algebra ℝ) (p : Parameters) (version : Bool) :
    (IO.real rest p version).Outputs ≃ (Fin p.N → Fin p.M → Int) where
  toFun := fun ys c r => ys ⟨0,by simp [IO.real,IO.grid]⟩ (r,c,())
  invFun := fun ys => Fin.cases (fun j => ys j.2.1 j.1) (fun k => k.elim0)
  left_inv := by intro ys; funext o j; fin_cases o; rfl
  right_inv := fun _ => rfl

/-- Full source grid and the actual unsigned low-eight-bit integer observation
are fixed in the type. No Real output or observer is substituted. -/
def transposeIO (rest : Algebra ℝ) (p : Parameters) (version : Bool) :
    LogicalIO (IO.real rest p version).Inputs (IO.real rest p version).Outputs
      (Inputs p) (Fin p.N → Fin p.M → Int)
      ((IO.real rest p version).ObservesInteger (IO.byteView (IO.real rest p version))) where
  inputs := inputEquiv rest p version
  outputs := outputEquiv rest p version
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.LogicalView

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.Reference
open VeriTile.Triton FP.Structural KernelSource KernelModels
noncomputable section

/-- The observable byte bits, represented in the unsigned interval 0..255.
Real arithmetic and nearest-even/clamped signed64 conversion are mathematical
semantics; they do not assert exact native arithmetic. -/
def realTarget (rest : Algebra ℝ) (p : Parameters) (version : Bool) :
    (IO.real rest p version).Inputs → (IO.real rest p version).Outputs :=
  IO.formula (realAlgebra rest) realRound p version
    (fun x inverse => ((realByte x inverse).toNat : Int))

end
end VeriTile.Bench.Repairs.QuantTransposeKernel.Reference
