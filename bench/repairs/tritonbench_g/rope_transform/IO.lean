import bench.repairs.tritonbench_g.rope_transform.Kernels
import bench.repairs.tritonbench_g.rope_transform.Proofs.Index
import VeriTile.Triton.Float.LogicalProgramIO

/-! Full forward rotation IO. Q/K are explicit in-place FP32 channels; C/S are
contiguous read-only tables. The selected geometry has even positive dimensions. -/
namespace VeriTile.Bench.Repairs.RopeTransform.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section

def inputs (r : Request) : List IOChannel :=
  [⟨.fp32, [r.rows,r.qHeads,r.dimension], "Q"⟩,
   ⟨.fp32, [r.rows,r.kHeads,r.dimension], "K"⟩,
   ⟨.fp32, [r.sequence,r.half], "C"⟩,
   ⟨.fp32, [r.sequence,r.half], "S"⟩]
def outputs (r : Request) : List IOChannel :=
  [⟨.fp32, [r.rows,r.qHeads,r.dimension], "Q"⟩,
   ⟨.fp32, [r.rows,r.kHeads,r.dimension], "K"⟩]
abbrev Inputs (A : Type) (r : Request) := ChannelValues A (inputs r)

/-- Logical values extended outside the declared finite windows only for
stating scalar formulas. No execution or raw memory occurs in these views. -/
def q [Inhabited A] {r : Request} (xs : Inputs A r)
    (key : Bool) (row head col : Nat) : A :=
  if hr : row<r.rows then if hc : col<r.dimension then
    if key then
      if hh : head<r.kHeads then xs ⟨1, by simp [inputs]⟩
        (⟨row,hr⟩,⟨head,hh⟩,⟨col,hc⟩,PUnit.unit) else default
    else
      if hh : head<r.qHeads then xs ⟨0, by simp [inputs]⟩
        (⟨row,hr⟩,⟨head,hh⟩,⟨col,hc⟩,PUnit.unit) else default
  else default else default

def coefficient [Inhabited A] {r : Request} (xs : Inputs A r)
    (sine : Bool) (row lane : Nat) : A :=
  if hr : row<r.sequence then if hl : lane<r.half then
    if sine then xs ⟨3, by simp [inputs]⟩ (⟨row,hr⟩,⟨lane,hl⟩,PUnit.unit)
    else xs ⟨2, by simp [inputs]⟩ (⟨row,hr⟩,⟨lane,hl⟩,PUnit.unit)
  else default else default

def result {r : Request} (f : Bool → Nat → Nat → Nat → A) :
    ChannelValues A (outputs r)
  | ⟨0,_⟩ => fun j => f Bool.false j.1.val j.2.1.val j.2.2.1.val
  | ⟨1,_⟩ => fun j => f Bool.true j.1.val j.2.1.val j.2.2.1.val
  | ⟨k+2,h⟩ => False.elim (by simp [outputs] at h)

def invocation {β : Type} (M : Algebra β) (r : Request) (row : Nat) : Invocation β where
  kernel := kernel r
  backend := M
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then row else 0
  numPids := some (fun ax => if ax=0 then r.rows else 1)

/-- The reused typed proof source is exactly the selected shared source,
including the actual forward branch and all production padding parameters. -/
theorem source_identity {β : Type} (M : Algebra β) (r : Request) (row : Nat) :
    (invocation M r row).kernel =
      Kernels.source .fp32 .fp32 "Q" "K" "C" "S"
        (r.stride Bool.false) (r.stride Bool.true) r.half r.half r.sequence r.batches
        r.qHeads r.kHeads r.dimension (r.paddedHeads Bool.false) (r.paddedHeads Bool.true)
        r.paddedDimension (max (r.paddedHeads Bool.false) (r.paddedHeads Bool.true)) Bool.false := rfl

/-- The complete selected FP32 grid, including its empty-grid behavior.
Input/output Q/K region names coincide, making the in-place footprint explicit. -/
@[reducible] def sourceIO {α β : Type} (M : Algebra β) (encode : α → β)
    (observe : β → α) (r : Request) : ProgramIO α β where
  inputs := inputs r
  outputs := outputs r
  scratch := []
  buffers := ["Q","K","C","S"]
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun j => r.offset Bool.false j.1.val j.2.1.val j.2.2.1.val
    | ⟨1,_⟩ => fun j => r.offset Bool.true j.1.val j.2.1.val j.2.2.1.val
    | ⟨2,_⟩ | ⟨3,_⟩ => fun j => j.1.val*r.half+j.2.1.val
    | ⟨k+4,h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun _ => True
    | ⟨2,_⟩ | ⟨3,_⟩ => fun _ => 0<r.rows
    | ⟨k+4,h⟩ => False.elim (by simp [inputs] at h)
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun j => r.offset Bool.false j.1.val j.2.1.val j.2.2.1.val
    | ⟨1,_⟩ => fun j => r.offset Bool.true j.1.val j.2.1.val j.2.2.1.val
    | ⟨k+2,h⟩ => False.elim (by simp [outputs] at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (List.range r.rows).map (invocation M r)
  encode := encode
  observe := fun _ _ => observe

/-- Source-independent logical rotation formula, parameterized only by its
four-operand scalar mathematics or exact native primitive expression. -/
def target [Inhabited α] (pair : α → α → α → α → Bool → α)
    (r : Request) (xs : Inputs α r) : ChannelValues α (outputs r) :=
  result (fun key row head col =>
    let lane := if col<r.half then col else col-r.half
    pair (q xs key row head lane) (q xs key row head (lane+r.half))
      (coefficient xs Bool.false (row%r.sequence) lane)
      (coefficient xs Bool.true (row%r.sequence) lane) (decide (r.half≤col)))

#specNonCircular target avoiding [kernel,Kernels.source,sourceIO]
#axiomsClean source_identity
end
end VeriTile.Bench.Repairs.RopeTransform.PublicIO

namespace VeriTile.Bench.Repairs.RopeTransform.LogicalIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- All four source arrays, on logical coordinates, including inactive inputs. -/
structure Inputs (r : Request) where
  query : Fin r.rows → Fin r.qHeads → Fin r.dimension → ℝ
  key : Fin r.rows → Fin r.kHeads → Fin r.dimension → ℝ
  cosine : Fin r.sequence → Fin r.half → ℝ
  sine : Fin r.sequence → Fin r.half → ℝ

/-- Both complete in-place result tensors. -/
structure Outputs (r : Request) where
  query : Fin r.rows → Fin r.qHeads → Fin r.dimension → ℝ
  key : Fin r.rows → Fin r.kHeads → Fin r.dimension → ℝ

def inputEquiv (r : Request) : PublicIO.Inputs ℝ r ≃ Inputs r where
  toFun := fun xs => ⟨fun row head col => xs (0 : Fin 4) (row,head,col,()),
    fun row head col => xs (1 : Fin 4) (row,head,col,()),
    fun row lane => xs (2 : Fin 4) (row,lane,()), fun row lane => xs (3 : Fin 4) (row,lane,())⟩
  invFun := fun xs => Fin.cases (fun j => xs.query j.1 j.2.1 j.2.2.1)
    (Fin.cases (fun j => xs.key j.1 j.2.1 j.2.2.1)
      (Fin.cases (fun j => xs.cosine j.1 j.2.1)
        (Fin.cases (fun j => xs.sine j.1 j.2.1) (fun i => i.elim0))))
  left_inv := by intro xs; funext port j; fin_cases port <;> cases j <;> rfl
  right_inv := by intro xs; cases xs; rfl

def outputEquiv (r : Request) : ChannelValues ℝ (PublicIO.outputs r) ≃ Outputs r where
  toFun := fun xs => ⟨fun row head col => xs (0 : Fin 2) (row,head,col,()),
    fun row head col => xs (1 : Fin 2) (row,head,col,())⟩
  invFun := fun xs => Fin.cases (fun j => xs.query j.1 j.2.1 j.2.2.1)
    (Fin.cases (fun j => xs.key j.1 j.2.1 j.2.2.1) (fun i => i.elim0))
  left_inv := by intro xs; funext port j; fin_cases port <;> cases j <;> rfl
  right_inv := by intro xs; cases xs; rfl

/-- Total logical extensions; no channel, memory address or execution occurs. -/
def query {r : Request} (xs : Inputs r) (key : Bool) (row head col : Nat) : ℝ :=
  if hr : row<r.rows then if hc : col<r.dimension then
    if key then
      if hh : head<r.kHeads then xs.key ⟨row,hr⟩ ⟨head,hh⟩ ⟨col,hc⟩ else 0
    else
      if hh : head<r.qHeads then xs.query ⟨row,hr⟩ ⟨head,hh⟩ ⟨col,hc⟩ else 0
  else 0 else 0

def coefficient {r : Request} (xs : Inputs r) (sine : Bool) (row lane : Nat) : ℝ :=
  if hr : row<r.sequence then if hl : lane<r.half then
    if sine then xs.sine ⟨row,hr⟩ ⟨lane,hl⟩ else xs.cosine ⟨row,hr⟩ ⟨lane,hl⟩
  else 0 else 0

/-- The complete forward grid; only its mathematical coordinates change. -/
def view (M : Algebra ℝ) (r : Request) : LogicalProgramIO ℝ ℝ (Inputs r) (Outputs r) where
  core := PublicIO.sourceIO M id id r
  inputs := inputEquiv r
  outputs := outputEquiv r

end
end VeriTile.Bench.Repairs.RopeTransform.LogicalIO
