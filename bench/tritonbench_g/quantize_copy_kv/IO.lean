import bench.tritonbench_g.quantize_copy_kv.Proofs.RawKernel
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.LogicalProgramIO
import Mathlib.Tactic.FinCases

/-! Complete quantized-copy IO: signed destination metadata and FP32 data,
INT8 codes and FP16 scales. Both versions retain the same full token schedule
and differ only in the displayed integer address multiplication. -/
namespace VeriTile.Bench.Repairs.QuantizeCopyKv.Interface
open VeriTile.Triton FP.Structural KernelSource
noncomputable section

def inputs (p : Parameters) (count : Nat) : List IOChannel :=
  [⟨.int, [count], "destination"⟩, ⟨.real, [count, p.heads, p.width], "input"⟩]
def outputs (p : Parameters) (count : Nat) : List IOChannel :=
  [⟨.int, [count, p.heads, p.width], "output"⟩, ⟨.fp16, [count, p.heads], "scale"⟩]
abbrev Inputs (A : Type) (p : Parameters) (count : Nat) := ChannelValues A (inputs p count)
abbrev Outputs (A : Type) (p : Parameters) (count : Nat) := ChannelValues A (outputs p count)

def read (p : Parameters) (count : Nat) (i : Fin (inputs p count).length) :
    TileIndex (inputs p count)[i].shape → Nat :=
  match i with
  | ⟨0, _⟩ => fun j => j.1.val
  | ⟨1, _⟩ => fun j => KernelResult.inputOffset p j.1.val j.2.1.val j.2.2.1.val
  | ⟨n + 2, h⟩ => False.elim (by simp [inputs] at h)
def write (p : Parameters) (count : Nat) (dest : Nat → Nat) (o : Fin (outputs p count).length) :
    TileIndex (outputs p count)[o].shape → Nat :=
  match o with
  | ⟨0, _⟩ => fun j => KernelResult.codeOffset p (dest j.1.val) j.2.1.val j.2.2.1.val
  | ⟨1, _⟩ => fun j => KernelResult.scaleOffset p (dest j.1.val) j.2.1.val
  | ⟨n + 2, h⟩ => False.elim (by simp [outputs] at h)

def memory (A : Type) (p : Parameters) (count : Nat) (dest : Nat → Nat) : MemoryIO A where
  inputs := inputs p count
  outputs := outputs p count
  scratch := []
  buffers := ["destination", "input", "output", "scale"]
  read := fun i _ => read p count i
  inputMask := fun _ _ _ => True
  write := fun o _ => write p count dest o
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i

def invocation (M : Algebra A) (convert : A → Int) (equal : A → A → Bool)
    (p : Parameters) (version : Bool) (token : Nat) : Invocation A where
  kernel := kernel version p
  backend := KernelExecution.backend M convert equal
  pids := fun _ => token
  setPids := fun ax => decide (ax = 0)

def programIO (M : Algebra A) (convert : A → Int) (equal : A → A → Bool)
    (p : Parameters) (count : Nat) (dest : Nat → Nat) (version : Bool) : ProgramIO A A :=
  { memory A p count dest with
    schedule := fun _ => (List.range count).map (invocation M convert equal p version)
    encode := id
    observe := fun _ _ => id }

def candidate (M : Algebra A) (convert : A → Int) (equal : A → A → Bool)
    (p : Parameters) (count : Nat) (dest : Nat → Nat) :
    ProgramIO.Variant (programIO M convert equal p count dest Bool.false) where
  schedule := fun _ => (List.range count).map (invocation M convert equal p Bool.true)
  observe := fun _ _ => id

/-- Metadata is the signed representation of the same logical destinations
used by the public output windows. This is the existing input-binding premise. -/
def metadataMatches {A : Type} (p : Parameters) (count : Nat) (dest : Nat → Nat)
    (values : Inputs A p count) : Prop :=
  ∀ t : Fin count, values ⟨0, by simp [inputs]⟩ (t, PUnit.unit) = (dest t.val : Int)

def payload {A : Type} [Inhabited A] {p : Parameters} {count : Nat}
    (values : Inputs A p count) (token head column : Nat) : A :=
  if ht : token < count then if hh : head < p.heads then if hc : column < p.width then
    values ⟨1, by simp [inputs]⟩ (⟨token, ht⟩, ⟨head, hh⟩, ⟨column, hc⟩, PUnit.unit)
  else default else default else default

def result {A : Type} {p : Parameters} {count : Nat}
    (codes : Nat → Nat → Nat → Int) (scales : Nat → Nat → A) : Outputs A p count :=
  fun o => match o with
  | ⟨0, _⟩ => fun j => codes j.1.val j.2.1.val j.2.2.1.val
  | ⟨1, _⟩ => fun j => scales j.1.val j.2.1.val
  | ⟨n + 2, h⟩ => False.elim (by simp [outputs] at h)

end
end VeriTile.Bench.Repairs.QuantizeCopyKv.Interface

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.LogicalView
open VeriTile.Triton FP.Structural KernelSource
noncomputable section

/-- Named mathematical tensor and signed destination metadata. -/
structure Inputs (p : Parameters) (count : Nat) where
  values : Fin count → Fin p.heads → Fin (p.width) → ℝ
  destination : Fin count → Int

/-- Both actual outputs: signed INT8 codes and the source scale channel. -/
structure Outputs (p : Parameters) (count : Nat) where
  code : Fin count → Fin p.heads → Fin (p.width) → Int
  scale : Fin count → Fin p.heads → ℝ

def inputEquiv (p : Parameters) (count : Nat) : Interface.Inputs ℝ p count ≃ Inputs p count where
  toFun := fun xs => ⟨(fun t h c => xs ⟨1,by simp [Interface.inputs]⟩ (t,h,c,())),fun t => xs ⟨0,by simp [Interface.inputs]⟩ (t,())⟩
  invFun := fun xs => Fin.cases (fun j => xs.destination j.1) (Fin.cases (fun j => xs.values j.1 j.2.1 j.2.2.1) (fun k => k.elim0))
  left_inv := by
    intro xs; funext o j; fin_cases o <;> rfl
  right_inv := by intro xs; cases xs; rfl

def outputEquiv (p : Parameters) (count : Nat) : ChannelValues ℝ (Interface.outputs p count) ≃ Outputs p count where
  toFun := fun ys => ⟨(fun t h c => ys ⟨0,by simp [Interface.outputs]⟩ (t,h,c,())),fun t h => ys ⟨1,by simp [Interface.outputs]⟩ (t,h,())⟩
  invFun := fun ys => Fin.cases (fun j => ys.code j.1 j.2.1 j.2.2.1)
    (Fin.cases (fun j => ys.scale j.1 j.2.1) (fun k => k.elim0))
  left_inv := by intro ys; funext o j; fin_cases o <;> rfl
  right_inv := by intro ys; cases ys; rfl

/-- Independent zero extension of a finite mathematical row. -/
def row {p : Parameters} {count : Nat} (xs : Inputs p count)
    (t : Fin count) (h : Fin p.heads) (c : Nat) : ℝ :=
  if hc : c<p.width then xs.values t h ⟨c,hc⟩ else 0

def quantizeIO (rest : Algebra ℝ) (p : Parameters) (count : Nat)
    (dest : Nat → Nat) (version : Bool) : LogicalProgramIO ℝ ℝ (Inputs p count) (Outputs p count) where
  core := Interface.programIO (KernelRealModel.algebra rest) KernelRealModel.convert KernelRealModel.equal p count dest version
  inputs := inputEquiv p count
  outputs := outputEquiv p count

def Metadata (dest : Nat → Nat) {p : Parameters} {count : Nat} (xs : Inputs p count) : Prop :=
  ∀ t, xs.destination t=(dest t.val : Int)
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.LogicalView
