import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.fast_rope_embedding.Proofs.KernelInterface
import VeriTile.Triton.Float.ProgramIO

/-! The shared FP32 rotary IO uses an in-place Q tile and separate cosine/sine
rows. Only the active four-head group is written; odd trailing columns remain
outside the output channel. All source direction and stride arguments remain
visible in the actual invocation. -/
namespace VeriTile.Bench.Repairs.FastRopeEmbedding.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section

def inputs (r : Request) : List IOChannel :=
  [⟨.real, [r.heads, 2 * r.half], "Q"⟩, ⟨.real, [r.half], "C"⟩, ⟨.real, [r.half], "S"⟩]
def outputs (r : Request) : List IOChannel := [⟨.real, [r.heads, 2 * r.half], "Q"⟩]
abbrev Inputs (A : Type) (r : Request) := ChannelValues A (inputs r)
abbrev Outputs (A : Type) (r : Request) := ChannelValues A (outputs r)

def read (r : Request) (row : Nat) (i : Fin (inputs r).length) :
    TileIndex (inputs r)[i].shape → Nat :=
  match i with
  | ⟨0, _⟩ => fun j => r.offset row j.1.val j.2.1.val
  | ⟨1, _⟩ => fun j => row % r.sequence * r.half + j.1.val
  | ⟨2, _⟩ => fun j => row % r.sequence * r.half + j.1.val
  | ⟨n + 3, h⟩ => False.elim (by simp [inputs] at h)
def inputMask (r : Request) (group : Nat) (i : Fin (inputs r).length) :
    TileIndex (inputs r)[i].shape → Prop :=
  match i with
  | ⟨0, _⟩ => fun j => group * 4 ≤ j.1.val ∧ j.1.val < min (group * 4 + 4) r.heads
  | ⟨1, _⟩ | ⟨2, _⟩ => fun _ => True
  | ⟨n + 3, h⟩ => False.elim (by simp [inputs] at h)
def write (r : Request) (row : Nat) (i : Fin (outputs r).length) :
    TileIndex (outputs r)[i].shape → Nat :=
  match i with
  | ⟨0, _⟩ => fun j => r.offset row j.1.val j.2.1.val
  | ⟨n + 1, h⟩ => False.elim (by simp [outputs] at h)
def outputMask (r : Request) (group : Nat) (i : Fin (outputs r).length) :
    TileIndex (outputs r)[i].shape → Prop :=
  match i with
  | ⟨0, _⟩ => fun j => group * 4 ≤ j.1.val ∧ j.1.val < min (group * 4 + 4) r.heads
  | ⟨n + 1, h⟩ => False.elim (by simp [outputs] at h)

def memory (A : Type) (r : Request) (row group : Nat) : MemoryIO A where
  inputs := inputs r
  outputs := outputs r
  scratch := []
  buffers := ["Q", "C", "S"]
  read := fun i _ => read r row i
  inputMask := fun i _ => inputMask r group i
  write := fun i _ => write r row i
  outputMask := fun i _ => outputMask r group i
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i

def invocation (op : Operations A) (r : Request) (qStride cosineStride sineStride : Nat)
    (backward : Bool) (row group : Nat) : Invocation A where
  kernel := KernelInterface.rawSource r qStride cosineStride sineStride backward
  backend := op.backend
  pids := fun ax => if ax = 0 then row else if ax = 1 then group else 0

def programIO (op : Operations A) (r : Request) (qStride cosineStride sineStride : Nat)
    (backward : Bool) (row group : Nat) : ProgramIO A A :=
  { memory A r row group with
    schedule := fun _ => [invocation op r qStride cosineStride sineStride backward row group]
    encode := id
    observe := fun _ _ => id }

def query {A : Type} {r : Request} (values : Inputs A r)
    (h : Fin r.heads) (c : Fin (2 * r.half)) : A :=
  values ⟨0, by simp [inputs]⟩ (h, c, PUnit.unit)
def cosine {A : Type} {r : Request} (values : Inputs A r) (i : Fin r.half) : A :=
  values ⟨1, by simp [inputs]⟩ (i, PUnit.unit)
def sine {A : Type} {r : Request} (values : Inputs A r) (i : Fin r.half) : A :=
  values ⟨2, by simp [inputs]⟩ (i, PUnit.unit)
def lane (r : Request) (c : Fin (2 * r.half)) : Fin r.half :=
  ⟨if c.val < r.half then c.val else c.val - r.half, by have := c.isLt; split <;> omega⟩
def left (r : Request) (i : Fin r.half) : Fin (2 * r.half) := ⟨i.val, by have := i.isLt; omega⟩
def right (r : Request) (i : Fin r.half) : Fin (2 * r.half) := ⟨i.val + r.half, by have := i.isLt; omega⟩
def result {A : Type} {r : Request} (f : Fin r.heads → Fin (2 * r.half) → A) : Outputs A r :=
  fun o => match o with
  | ⟨0, _⟩ => fun j => f j.1 j.2.1
  | ⟨n + 1, h⟩ => False.elim (by simp [outputs] at h)

end
end VeriTile.Bench.Repairs.FastRopeEmbedding.Interface

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.LogicalIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section

/-- All source inputs, with logical head/coordinate and coefficient indices. -/
structure Inputs (r : Request) where
  query : Fin r.heads → Fin (2*r.half) → ℝ
  cosine : Fin r.half → ℝ
  sine : Fin r.half → ℝ

def inputEquiv (r : Request) : Interface.Inputs ℝ r ≃ Inputs r where
  toFun xs := ⟨Interface.query xs,Interface.cosine xs,Interface.sine xs⟩
  invFun xs := Fin.cases (fun j => xs.query j.1 j.2.1)
    (Fin.cases (fun j => xs.cosine j.1) (Fin.cases (fun j => xs.sine j.1) (fun i => i.elim0)))
  left_inv := by
    intro xs
    funext port j
    fin_cases port <;> cases j <;> rfl
  right_inv := by intro xs; cases xs; rfl

def outputEquiv (r : Request) : Interface.Outputs ℝ r ≃ (Fin r.heads → Fin (2*r.half) → ℝ) where
  toFun ys h c := ys (0 : Fin 1) (h,c,())
  invFun := Interface.result
  left_inv := by intro ys; funext port j; fin_cases port; cases j; rfl
  right_inv := by intro ys; rfl


def view (r : Request) (qStride cosineStride sineStride : Nat)
    (backward : Bool) (row group : Nat) :
    LogicalProgramIO ℝ ℝ (Inputs r) (Fin r.heads → Fin (2*r.half) → ℝ) where
  core := Interface.programIO ideal r qStride cosineStride sineStride backward row group
  inputs := inputEquiv r
  outputs := outputEquiv r

end
end VeriTile.Bench.Repairs.FastRopeEmbedding.LogicalIO
