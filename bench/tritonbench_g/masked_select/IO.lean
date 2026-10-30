import bench.tritonbench_g.masked_select.Proofs.Kernel
import VeriTile.Triton.Float.InputBindingIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Float.LogicalProgramIO


/-! Public kernel input/output channels. Prepared prefix sums are explicit
inputs, with their inclusive-count meaning stated as a logical domain. The
kernel source remains FP32/Bool/Int64; Nat is the proven nonnegative prefix
projection, and `.real` is the source interpreter's FP32 payload channel. -/
namespace VeriTile.Bench.Repairs.MaskedSelect.IO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MaskedSelect TypedCopy Whole StableSelection
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (copyBackend)
noncomputable section

abbrev channels (p : Parameters) : List IOChannel :=
  [⟨.real, [p.size], p.input⟩, ⟨.bool, [p.size], p.selection.cast⟩,
   ⟨.nat, [p.size], p.prefixSum.cast⟩]
abbrev Inputs (α : Type) (p : Parameters) := ChannelValues α (channels p)

def payload {α : Type} [Inhabited α] {p : Parameters} (vals : Inputs α p) (i : Nat) : α :=
  if h : i < p.size then vals 0 (⟨i, h⟩, PUnit.unit) else default

def mask {α : Type} {p : Parameters} (vals : Inputs α p) (i : Nat) : Bool :=
  if h : i < p.size then vals 1 (⟨i, h⟩, PUnit.unit) else Bool.false

/-- An input relation, independent of the source execution or output cells. -/
def Prefixes {α : Type} (p : Parameters) (vals : Inputs α p) : Prop :=
  ∀ i : Fin p.size, vals 2 (i, PUnit.unit) = count (mask vals) (i.val + 1)

/-- The actual selected source and complete one-dimensional launch ABI. -/
def launch {α : Type} (rest : Algebra α) (p : Parameters) (pid : Nat) : Invocation α where
  kernel := kernel p
  backend := copyBackend rest
  pids := fun a => if a = 0 then pid else 0
  numPids := some (fun a => if a = 0 then grid p.size p.width else 1)

def schedule {α : Type} (rest : Algebra α) (p : Parameters) (config : Config) :
    List (Invocation α) :=
  (List.range (grid p.size config.width)).map (launch rest {p with width := config.width})

/-- Logical output indices have capacity n and active extent count(mask,n),
exactly the stable selected sequence. There is no scratch or hidden output. -/
abbrev selectionIO {α : Type} (rest : Algebra α) (p : Parameters) (config : Config) :
    ProgramIO α α where
  inputs := channels p
  outputs := [⟨.real, [p.size], p.output⟩]
  scratch := []
  buffers := [p.input, p.selection.cast, p.prefixSum.cast, p.output]
  read := Fin.cases (fun _ j => j.1.val)
    (Fin.cases (fun _ j => j.1.val) (Fin.cases (fun _ j => j.1.val) (fun i => i.elim0)))
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val) (fun i => i.elim0)
  outputMask := Fin.cases (fun vals j => j.1.val < count (mask vals) p.size) (fun i => i.elim0)
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => schedule rest p config
  encode := id
  observe := fun _ _ => id

def candidate {α : Type} (rest : Algebra α) (p : Parameters) (left right : Config) :
    ProgramIO.Variant (selectionIO rest p left) where
  schedule := fun _ => schedule rest p right
  observe := fun _ _ => id

/-- Complete configuration schedules on the original exact input cells and
the independent inclusive-prefix domain. -/
def original {α : Type} (rest : Algebra α) (p : Parameters) (left : Config) :=
  ProgramIO.Implementation.original (selectionIO rest p left) (Prefixes p) .exactCells

def configured {α : Type} (rest : Algebra α) (p : Parameters) (left right : Config) :=
  ProgramIO.Implementation.ofVariant (candidate rest p left right) (Prefixes p) .exactCells

/-- Stable selection in input order, independent of block size and scatter. -/
def selected {α : Type} [Inhabited α] (p : Parameters) (vals : Inputs α p) : List α :=
  ((List.range p.size).filter (mask vals)).map (payload vals)

def mathematical {α : Type} [Inhabited α] (p : Parameters) (vals : Inputs α p) :
    ChannelValues α [⟨.real, [p.size], p.output⟩] :=
  Fin.cases (fun j => (selected p vals)[j.1.val]?.getD default) (fun i => i.elim0)

#specNonCircular selected avoiding [kernel, Kernel.run]
end
end VeriTile.Bench.Repairs.MaskedSelect.IO

namespace VeriTile.Bench.Repairs.MaskedSelect.ReaderIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MaskedSelect TypedCopy Whole StableSelection
noncomputable section

/-- Payload, Boolean selection and the prepared inclusive prefix are all inputs. -/
structure Inputs (p : Parameters) where
  payload : Fin p.size → ℝ
  mask : Fin p.size → Bool
  inclusive : Fin p.size → Nat

def inputs (p : Parameters) : IO.Inputs ℝ p ≃ Inputs p where
  toFun := fun xs => ⟨(fun j => xs (0 : Fin 3) (j,())),
    (fun j => xs (1 : Fin 3) (j,())), (fun j => xs (2 : Fin 3) (j,()))⟩
  invFun := fun xs => Fin.cases (fun j => xs.payload j.1)
    (Fin.cases (fun j => xs.mask j.1) (Fin.cases (fun j => xs.inclusive j.1) (fun i => i.elim0)))
  left_inv := by intro xs; funext i j; fin_cases i <;> rcases j with ⟨j,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

def Inputs.value {p : Parameters} (xs : Inputs p) (i : Nat) : ℝ :=
  if h : i<p.size then xs.payload ⟨i,h⟩ else 0

def Inputs.selected {p : Parameters} (xs : Inputs p) (i : Nat) : Bool :=
  if h : i<p.size then xs.mask ⟨i,h⟩ else Bool.false

/-- Exactly the original prepared prefix relation; wrapper preparation is external. -/
def prefixes (p : Parameters) (xs : Inputs p) : Prop :=
  ∀ i : Fin p.size, xs.inclusive i = count xs.selected (i.val+1)

/-- Capacity remains n; the original IO exposes only count(mask,n) active outputs. -/
def view (rest : Algebra ℝ) (p : Parameters) (config : Config) :
    LogicalDomainIO (IO.selectionIO rest p config).Inputs (IO.selectionIO rest p config).Outputs
      (Inputs p) (Fin p.size → ℝ) ((IO.selectionIO rest p config).ImplementsOnCells) where
  inputs := inputs p
  outputs := (ChannelValues.singleEquiv ℝ _).trans (Values.vectorEquiv ℝ .real p.size)
end
end VeriTile.Bench.Repairs.MaskedSelect.ReaderIO
