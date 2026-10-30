import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.RawInterface
import VeriTile.Triton.Float.ObservedProgramIO

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.PhiIO
open VeriTile.Triton FP.Structural FlashDecode2
noncomputable section

/-- Source scalar geometry and a separate logical read-window extent. `capacity`
does not replace source stride `S`: arbitrary dynamic counts, including counts
larger than S, remain representable by choosing a sufficiently large window. -/
structure Geometry where
  B : Nat
  H : Nat
  S : Nat
  D : Nat
  Q : Nat
  BD : Nat
  capacity : Nat

abbrev channels (g : Geometry) : List IOChannel :=
  [⟨.real,[g.B*g.H,g.capacity],"E"⟩,
   ⟨.real,[g.B*g.H,g.capacity,g.D],"M"⟩,⟨.nat,[g.B],"L"⟩]
abbrev outputs (g : Geometry) : List IOChannel := [⟨.real,[g.B*g.H,g.D],"O"⟩]
abbrev Inputs (α : Type) (g : Geometry) := ChannelValues α (channels g)

def lengths {α : Type} {g : Geometry} (xs : Inputs α g) (b : Nat) : Nat :=
  if h : b<g.B then xs 2 (⟨b,h⟩,PUnit.unit) else 0

/-- Exact integer source count, including zero length and zero divisor. -/
def iterations (Q length : Nat) : Nat := (if length≤0 then 0 else length+Q-1)/Q
def steps {α : Type} {g : Geometry} (xs : Inputs α g) (row : Nat) : Nat :=
  iterations g.Q (lengths xs (row/g.H))

/-- Backing-window coverage, independent of the source's row stride S. -/
def Fits {α : Type} (g : Geometry) (xs : Inputs α g) : Prop :=
  ∀ row,row<g.B*g.H → steps xs row≤g.capacity
def Positive {α : Type} (g : Geometry) (xs : Inputs α g) : Prop :=
  ∀ row,row<g.B*g.H → 0<steps xs row

def logs {α : Type} [Inhabited α] {g : Geometry} (xs : Inputs α g) (row j : Nat) : α :=
  if h : row<g.B*g.H ∧ j<g.capacity then xs 0 (⟨row,h.1⟩,⟨j,h.2⟩,PUnit.unit) else default
def mids {α : Type} [Inhabited α] {g : Geometry} (xs : Inputs α g) (row j d : Nat) : α :=
  if h : row<g.B*g.H ∧ j<g.capacity ∧ d<g.D then
    xs 1 (⟨row,h.1⟩,⟨j,h.2.1⟩,⟨d,h.2.2⟩,PUnit.unit) else default

def launch {β : Type} (A : Algebra β) (g : Geometry) (row : Nat) : Invocation β where
  kernel := phi g.H g.S g.D g.Q g.BD
  backend := A
  clearRegisters := Bool.true
  pids := fun a => if a=0 then row/g.H else if a=1 then row%g.H else 0
  numPids := some (fun a => if a=0 then g.B else if a=1 then g.H else 1)
def schedule {β : Type} (A : Algebra β) (g : Geometry) (order : List Nat) :=
  order.map (launch A g)

/-- Actual masked source, complete permuted B×H grid and source FP32
interpretation. E/M use their actual S strides; only dynamically used summary
windows are bound. L is natural metadata, O covers every active output, and
there is no writable scratch or input/output alias. -/
abbrev decodeIO {α β : Type} (A : Algebra β) (g : Geometry) (order : List Nat)
    (encode : α→β) (observe : β→α) : ProgramIO α β where
  inputs := channels g
  outputs := outputs g
  buffers := ["E","M","L","O"]
  scratch := []
  read := Fin.cases (fun _ j => j.1.val*g.S+j.2.1.val)
    (Fin.cases (fun _ j => j.1.val*g.S*g.D+j.2.1.val*g.D+j.2.2.1.val)
      (Fin.cases (fun _ j => j.1.val) (fun i => i.elim0)))
  inputMask := Fin.cases (fun xs j => j.2.1.val<steps xs j.1.val)
    (Fin.cases (fun xs j => j.2.1.val<steps xs j.1.val)
      (Fin.cases (fun _ _ => True) (fun i => i.elim0)))
  write := Fin.cases (fun _ j => j.1.val*g.D+j.2.1.val) (fun i => i.elim0)
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => schedule A g order
  encode := encode
  observe := fun _ _ => observe

def result {α : Type} {g : Geometry} (f : Fin (g.B*g.H) → Fin g.D → α) : ChannelValues α (outputs g) :=
  Fin.cases (fun j => f j.1 j.2.1) (fun i => i.elim0)

def mathematical (g : Geometry) (xs : Inputs ℝ g) :=
  result (g:=g) (fun row d => Reference.target (logs xs row.val)
    (mids xs row.val) (steps xs row.val) d.val)

/-- Independent native graph on arbitrary original symbolic input terms.
Evaluation is applied only to the final graph, retaining all compiler choices. -/
def nativeTerm (g : Geometry) (xs : Inputs Native.Term g) (row d : Nat) : Native.Term :=
  let a := KernelNative.scan (fun j => .load (logs xs row j))
    (fun j d => .load (if d<g.D then mids xs row j d else .literal (some .fp32) .real 0))
    (steps xs row)
  KernelNative.op .div (a.numerator d) a.denominator

#specNonCircular mathematical avoiding [Grid.source,FP.Structural.exec,values,Reference.recurrence]
#specNonCircular nativeTerm avoiding [Grid.source,FP.Structural.exec,values,Native.algebra]
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.PhiIO

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.LogicalIO
open VeriTile.Triton FP.Structural FlashDecode2
noncomputable section

/-- Logical split-summary tables and batch lengths, with no physical offsets. -/
structure Inputs (g : PhiIO.Geometry) where
  logs : Fin (g.B*g.H) → Fin g.capacity → ℝ
  mids : Fin (g.B*g.H) → Fin g.capacity → Fin g.D → ℝ
  lengths : Fin g.B → Nat

def inputEquiv (g : PhiIO.Geometry) : PhiIO.Inputs ℝ g ≃ Inputs g where
  toFun xs := ⟨fun row j => xs (0 : Fin 3) (row,j,()),fun row j d => xs (1 : Fin 3) (row,j,d,()),fun b => xs (2 : Fin 3) (b,())⟩
  invFun xs := Fin.cases (fun j => xs.logs j.1 j.2.1)
    (Fin.cases (fun j => xs.mids j.1 j.2.1 j.2.2.1)
      (Fin.cases (fun j => xs.lengths j.1) (fun i => i.elim0)))
  left_inv := by intro xs; funext port j; fin_cases port <;> cases j <;> rfl
  right_inv := by intro xs; cases xs; rfl

def outputEquiv (g : PhiIO.Geometry) : ChannelValues ℝ (PhiIO.outputs g) ≃ (Fin (g.B*g.H) → Fin g.D → ℝ) where
  toFun ys row d := ys (0 : Fin 1) (row,d,())
  invFun ys := Fin.cases (fun j => ys j.1 j.2.1) (fun i => i.elim0)
  left_inv := by intro ys; funext port j; fin_cases port; cases j; rfl
  right_inv := by intro ys; rfl

/-- Total logical array access; only the declared prefix is used on the domain. -/
def Inputs.log {g : PhiIO.Geometry} (xs : Inputs g) (row : Fin (g.B*g.H)) (j : Nat) : ℝ :=
  if h : j<g.capacity then xs.logs row ⟨j,h⟩ else default

def Inputs.mid {g : PhiIO.Geometry} (xs : Inputs g) (row : Fin (g.B*g.H)) (j : Nat) (d : Fin g.D) : ℝ :=
  if h : j<g.capacity then xs.mids row ⟨j,h⟩ d else default

def Inputs.length {g : PhiIO.Geometry} (xs : Inputs g) (row : Fin (g.B*g.H)) : Nat :=
  if h : row.val/g.H<g.B then xs.lengths ⟨row.val/g.H,h⟩ else 0


def view (g : PhiIO.Geometry) (order : List Nat) (observe : Option ℝ → ℝ) :
    LogicalProgramIO ℝ (Option ℝ) (Inputs g) (Fin (g.B*g.H) → Fin g.D → ℝ) where
  core := PhiIO.decodeIO Reference.ideal g order some observe
  inputs := inputEquiv g
  outputs := outputEquiv g

def Inputs.steps {g : PhiIO.Geometry} (xs : Inputs g) (row : Fin (g.B*g.H)) : Nat :=
  PhiIO.iterations g.Q (xs.length row)
def Fits (g : PhiIO.Geometry) (xs : Inputs g) : Prop := ∀ row,xs.steps row≤g.capacity
def Positive (g : PhiIO.Geometry) (xs : Inputs g) : Prop := ∀ row,0<xs.steps row

end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.LogicalIO
