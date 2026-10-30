import bench.tritonbench_g.rope_backward_transform.Proofs.KernelInterface
import VeriTile.Triton.Float.ReadProgramIO
import VeriTile.Triton.Float.LogicalProgramIO

/-! Public complete-grid IO. Q and K are separate in-place channels; cosine
and sine retain independent row/column strides. Result reads include odd tails,
while the writable footprint contains only the two rotated halves. -/
namespace VeriTile.Bench.Repairs.RopeBackwardTransform.PublicIO
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

def invocation (op : Operations A) (r : Request) (row : Nat) : Invocation A where
  kernel := Source.kernel .fp32 .fp32 (parameters r)
  backend := op.backend
  pids := fun ax => if ax=0 then row else 0

/-- The existing grid list records completion order from last to first, so the
actual shared invocation schedule executes its reverse. All four FP32 stores
and both independently strided coefficient loads remain in each invocation. -/
def sourceIO (op : Operations A) (project : A → A) (r : Request) : ReadProgramIO A A where
  inputs := inputs r
  outputs := outputs r
  scratch := []
  buffers := ["Q","K","C","S"]
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun j => r.offset Bool.false j.1.val j.2.1.val j.2.2.1.val
    | ⟨1,_⟩ => fun j => r.offset Bool.true j.1.val j.2.1.val j.2.2.1.val
    | ⟨2,_⟩ => fun j => j.1.val*r.cosRowStride+j.2.1.val*r.cosColStride
    | ⟨3,_⟩ => fun j => j.1.val*r.sinRowStride+j.2.1.val*r.sinColStride
    | ⟨k+4,h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun _ => True
    | ⟨2,_⟩ | ⟨3,_⟩ => fun _ => KernelContract.launched r
    | ⟨k+4,h⟩ => False.elim (by simp [inputs] at h)
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun j => r.offset Bool.false j.1.val j.2.1.val j.2.2.1.val
    | ⟨1,_⟩ => fun j => r.offset Bool.true j.1.val j.2.1.val j.2.2.1.val
    | ⟨k+2,h⟩ => False.elim (by simp [outputs] at h)
  outputMask := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ => fun j => j.2.2.1.val<2*r.half
    | ⟨k+2,h⟩ => False.elim (by simp [outputs] at h)
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (KernelContract.launchRows r).reverse.map (invocation op r)
  encode := id
  observe := fun _ _ => project
  resultRead := fun port _ => match port with
    | ⟨0,_⟩ => fun j => r.offset Bool.false j.1.val j.2.1.val j.2.2.1.val
    | ⟨1,_⟩ => fun j => r.offset Bool.true j.1.val j.2.1.val j.2.2.1.val
    | ⟨k+2,h⟩ => False.elim (by simp [outputs] at h)
  resultMask := fun _ _ _ => True

end
end VeriTile.Bench.Repairs.RopeBackwardTransform.PublicIO

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.LogicalIO
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

/-- Preserve the complete ReadProgramIO result reads and its smaller writable
footprint, including the identity result at odd tails. -/
def view (op : Operations ℝ) (project : ℝ → ℝ) (r : Request) :
    VeriTile.Triton.LogicalIO (PublicIO.Inputs ℝ r) (ChannelValues ℝ (PublicIO.outputs r))
      (Inputs r) (Outputs r) (PublicIO.sourceIO op project r).Implements where
  inputs := inputEquiv r
  outputs := outputEquiv r

end
end VeriTile.Bench.Repairs.RopeBackwardTransform.LogicalIO
