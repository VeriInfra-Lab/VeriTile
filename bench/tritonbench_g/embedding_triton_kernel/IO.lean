import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.embedding_triton_kernel.Proofs.Execution
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.IO
open VeriTile.Triton FP.Structural
noncomputable section

/-- All possible vocabulary rows of the explicitly selected endpoint interval. -/
def vocabulary (p : Parameters) : Nat := (p.last-p.first).toNat

abbrev Ids := Fin 64 → Int
abbrev Weights (α : Type) (p : Parameters) := Fin (vocabulary p) → Fin p.width → α

/-- The int32 metadata channel and the FP32 weight-table channel. Structural
`.int`/`.real` are erased interpreter tags; storage precision is fixed by the quote. -/
def inputChannels (p : Parameters) : List IOChannel :=
  [⟨.int,[64],p.ids⟩,⟨.real,[vocabulary p,p.width],p.weight⟩]

def outputChannels (p : Parameters) : List IOChannel := [⟨.real,[64,p.width],p.output⟩]

def ids {α : Type} (p : Parameters) (xs : ChannelValues α (inputChannels p)) : Ids :=
  fun a => xs (0 : Fin 2) (a,PUnit.unit)

def weights {α : Type} (p : Parameters) (xs : ChannelValues α (inputChannels p)) : Weights α p :=
  fun k d => xs (1 : Fin 2) (k,d,PUnit.unit)

def active (p : Parameters) (pid : Nat) (a : Fin 64) : Prop := pid*64+a.val<p.n

def selected (p : Parameters) (pid : Nat) (token : Ids) (k : Fin (vocabulary p)) : Prop :=
  ∃ a : Fin 64, active p pid a ∧ p.first  ≤  token a ∧ token a<p.last ∧ (token a-p.first).toNat=k.val

/-- Exactly the signed-int32 metadata actually read by this invocation. -/
def signed {α : Type} (p : Parameters) (pid : Nat) (xs : ChannelValues α (inputChannels p)) : Prop :=
  ∀ a,active p pid a → -(2^31)  ≤  ids p xs a ∧ ids p xs a<2^31

/-- Actual complete 64-row source. The selected weight mask depends only on
logical metadata and mentions every vocabulary row that a live source load uses.
Unused weight rows carry neither input-binding nor storage-bound obligations. -/
def embeddingIO {α : Type} (rest : Algebra α) (zero : α)
    (candidate : Bool) (p : Parameters) (pid : Nat) : ProgramIO α α where
  inputs := inputChannels p
  outputs := outputChannels p
  scratch := []
  buffers := [p.ids,p.weight,p.output]
  read := by
    intro i xs
    refine Fin.cases ?_ ?_ (show Fin 2 from i)
    · exact fun j => pid*64+j.1.val
    · intro i
      have hi : i=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance i 0
      subst i
      exact fun j => j.1.val*p.weightStride+j.2.1.val
  inputMask := by
    intro i xs
    refine Fin.cases ?_ ?_ (show Fin 2 from i)
    · exact fun j => active p pid j.1
    · intro i
      have hi : i=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance i 0
      subst i
      exact fun j => selected p pid (ids p xs) j.1 ∧ j.2.1.val<p.hidden
  write := by
    intro i _
    have hi : i=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance i 0
    subst i
    exact fun j => (pid*64+j.1.val)*p.outputStride+j.2.1.val
  outputMask := by
    intro i _
    have hi : i=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance i 0
    subst i
    exact fun j => active p pid j.1 ∧ j.2.1.val<p.hidden
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => [{
    kernel := if candidate then indexCommutedSource .int32 p else source .int32 p
    backend := Execution.backend rest zero
    pids := fun ax => if ax=0 then pid else 0 }]
  encode := id
  observe := fun _ _ => id

/-- Connect an ordinary logical embedding formula to the declared channels. -/
def outputValues {α : Type} (p : Parameters)
    (f : Ids → Weights α p → Fin 64 → Fin p.width → α) :
    ChannelValues α (inputChannels p) → ChannelValues α (outputChannels p) := by
  intro xs i
  have hi : i=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance i 0
  subst i
  exact fun j => f (ids p xs) (weights p xs) j.1 j.2.1

end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.IO

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.LogicalIO
open VeriTile Triton FP.Structural
noncomputable section

structure Inputs (p : Parameters) where
  ids : Fin 64 → Int
  weights : Fin (IO.vocabulary p) → Fin p.width → ℝ

def inputEquiv (p : Parameters) : ChannelValues ℝ (IO.inputChannels p) ≃ Inputs p where
  toFun raw := ⟨IO.ids p raw,IO.weights p raw⟩
  invFun xs := Fin.cases (fun j => xs.ids j.1)
    (Fin.cases (fun j => xs.weights j.1 j.2.1) (fun i => i.elim0))
  left_inv := by intro raw; funext port j; fin_cases port <;> cases j <;> rfl
  right_inv := by intro xs; cases xs; rfl

def outputEquiv (p : Parameters) : ChannelValues ℝ (IO.outputChannels p) ≃ (Fin 64 → Fin p.width → ℝ) where
  toFun raw row col := raw (0 : Fin 1) (row,col,())
  invFun ys := Fin.cases (fun j => ys j.1 j.2.1) (fun i => i.elim0)
  left_inv := by intro raw; funext port j; fin_cases port; cases j; rfl
  right_inv := by intro ys; rfl

def view (rest : Algebra ℝ) (p : Parameters) (pid : Nat) :
    LogicalProgramIO ℝ ℝ (Inputs p) (Fin 64 → Fin p.width → ℝ) where
  core := IO.embeddingIO rest 0 Bool.false p pid
  inputs := inputEquiv p
  outputs := outputEquiv p

def Signed (p : Parameters) (pid : Nat) (xs : Inputs p) : Prop :=
  ∀ row,IO.active p pid row → -(2^31) ≤ xs.ids row ∧ xs.ids row<2^31

/-- Zero extension of the logical vocabulary table, independent of storage. -/
def Inputs.weightAt {p : Parameters} (xs : Inputs p) (row : Nat) (col : Fin p.width) : ℝ :=
  if h : row<IO.vocabulary p then xs.weights ⟨row,h⟩ col else 0

end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.LogicalIO
