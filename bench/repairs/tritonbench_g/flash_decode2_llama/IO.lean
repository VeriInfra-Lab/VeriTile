import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.RawKernel
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.IO
open VeriTile.Triton FP.Structural FlashDecode2 KernelInterface
noncomputable section

abbrev channels (g : KernelInterface.Geometry) : List IOChannel :=
  [⟨.real, [g.B * g.H, g.S], "E"⟩, ⟨.real, [g.B * g.H, g.S, g.D], "M"⟩,
   ⟨.nat, [g.B], "L"⟩]
abbrev Inputs (α : Type) (g : KernelInterface.Geometry) := ChannelValues α (channels g)

def logValues {α : Type} [Inhabited α] {g : KernelInterface.Geometry} (vals : Inputs α g)
    (row split : Nat) : α :=
  if h : row < g.B * g.H ∧ split < g.S then
    vals 0 (⟨row, h.1⟩, ⟨split, h.2⟩, PUnit.unit) else default

def midValues {α : Type} [Inhabited α] {g : KernelInterface.Geometry} (vals : Inputs α g)
    (row split dim : Nat) : α :=
  if h : row < g.B * g.H ∧ split < g.S ∧ dim < g.D then
    vals 1 (⟨row, h.1⟩, ⟨split, h.2.1⟩, ⟨dim, h.2.2⟩, PUnit.unit) else default

def lengths {α : Type} {g : KernelInterface.Geometry} (vals : Inputs α g) (batch : Nat) : Nat :=
  if h : batch < g.B then vals 2 (⟨batch, h⟩, PUnit.unit) else 0

/-- Existing length domain: only launched rows require positive lengths bounded
by the available split capacity. Empty batch/head grids remain allowed. -/
def Lengths {α : Type} (g : KernelInterface.Geometry) (vals : Inputs α g) : Prop :=
  ∀ row, row < g.B * g.H → 0 < lengths vals (row / g.H) ∧
    lengths vals (row / g.H) ≤ g.S * g.Q

def launch {β : Type} (A : Algebra β) (g : KernelInterface.Geometry) (row : Nat) : Invocation β where
  kernel := FlashDecode2.llama g.H g.S g.D g.Q g.D
  backend := A
  clearRegisters := Bool.true
  pids := fun a => if a = 0 then row / g.H else if a = 1 then row % g.H else 0
  numPids := some (fun a => if a = 0 then g.B else if a = 1 then g.H else 1)

def schedule {β : Type} (A : Algebra β) (g : KernelInterface.Geometry) (order : List Nat) :=
  order.map (launch A g)

/-- Full unmasked Llama source grid with natural length metadata and FP32
payload channels. Explicit order retains every original permutation. -/
abbrev decodeIO {α β : Type} (A : Algebra β) (g : KernelInterface.Geometry) (order : List Nat)
    (encode : α → β) (observe : β → α) : ProgramIO α β where
  inputs := channels g
  outputs := [⟨.real, [g.B * g.H, g.D], "O"⟩]
  scratch := []
  buffers := ["E", "M", "L", "O"]
  read := Fin.cases (fun _ j => j.1.val * g.S + j.2.1.val)
    (Fin.cases (fun _ j => j.1.val * g.S * g.D + j.2.1.val * g.D + j.2.2.1.val)
      (Fin.cases (fun _ j => j.1.val) (fun i => i.elim0)))
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val * g.D + j.2.1.val) (fun i => i.elim0)
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => schedule A g order
  encode := encode
  observe := fun _ _ => observe

def outputValues {α : Type} {g : KernelInterface.Geometry} (f : Fin (g.B * g.H) → Fin g.D → α) :
    ChannelValues α [⟨.real, [g.B * g.H, g.D], "O"⟩] :=
  Fin.cases (fun j => f j.1 j.2.1) (fun i => i.elim0)

/-- Independent exponentially weighted split summary. -/
def mathematical (g : KernelInterface.Geometry) (vals : Inputs ℝ g) :=
  outputValues (g := g) (fun row dim => Reference.target
    (logValues vals row.val) (midValues vals row.val)
    ((lengths vals (row.val / g.H) + g.Q - 1) / g.Q) dim.val)

#specNonCircular mathematical avoiding [Grid.source, FP.Structural.exec, values, Reference.recurrence]
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.IO

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.LogicalIO
open VeriTile.Triton FP.Structural FlashDecode2
noncomputable section

/-- Logical split-summary tables and batch lengths, with no physical offsets. -/
structure Inputs (g : KernelInterface.Geometry) where
  logs : Fin (g.B*g.H) → Fin g.S → ℝ
  mids : Fin (g.B*g.H) → Fin g.S → Fin g.D → ℝ
  lengths : Fin g.B → Nat

def inputEquiv (g : KernelInterface.Geometry) : IO.Inputs ℝ g ≃ Inputs g where
  toFun xs := ⟨fun row j => xs (0 : Fin 3) (row,j,()),fun row j d => xs (1 : Fin 3) (row,j,d,()),fun b => xs (2 : Fin 3) (b,())⟩
  invFun xs := Fin.cases (fun j => xs.logs j.1 j.2.1)
    (Fin.cases (fun j => xs.mids j.1 j.2.1 j.2.2.1)
      (Fin.cases (fun j => xs.lengths j.1) (fun i => i.elim0)))
  left_inv := by intro xs; funext port j; fin_cases port <;> cases j <;> rfl
  right_inv := by intro xs; cases xs; rfl

def outputEquiv (g : KernelInterface.Geometry) : ChannelValues ℝ [⟨.real,[g.B*g.H,g.D],"O"⟩] ≃ (Fin (g.B*g.H) → Fin g.D → ℝ) where
  toFun ys row d := ys (0 : Fin 1) (row,d,())
  invFun ys := Fin.cases (fun j => ys j.1 j.2.1) (fun i => i.elim0)
  left_inv := by intro ys; funext port j; fin_cases port; cases j; rfl
  right_inv := by intro ys; rfl

/-- Total logical array access; only the declared prefix is used on the domain. -/
def Inputs.log {g : KernelInterface.Geometry} (xs : Inputs g) (row : Fin (g.B*g.H)) (j : Nat) : ℝ :=
  if h : j<g.S then xs.logs row ⟨j,h⟩ else default

def Inputs.mid {g : KernelInterface.Geometry} (xs : Inputs g) (row : Fin (g.B*g.H)) (j : Nat) (d : Fin g.D) : ℝ :=
  if h : j<g.S then xs.mids row ⟨j,h⟩ d else default

def Inputs.length {g : KernelInterface.Geometry} (xs : Inputs g) (row : Fin (g.B*g.H)) : Nat :=
  if h : row.val/g.H<g.B then xs.lengths ⟨row.val/g.H,h⟩ else 0


def view (g : KernelInterface.Geometry) (order : List Nat) (observe : Option ℝ → ℝ) :
    LogicalProgramIO ℝ (Option ℝ) (Inputs g) (Fin (g.B*g.H) → Fin g.D → ℝ) where
  core := IO.decodeIO Reference.ideal g order some observe
  inputs := inputEquiv g
  outputs := outputEquiv g

def Lengths (g : KernelInterface.Geometry) (xs : Inputs g) : Prop :=
  ∀ row,0<xs.length row ∧ xs.length row≤g.S*g.Q

end
end VeriTile.Bench.Repairs.FlashDecode2Llama.LogicalIO
