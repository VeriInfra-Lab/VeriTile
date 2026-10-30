import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.token_attn_llama2.Proofs.Real
import bench.tritonbench_g.token_attn_llama2.Kernels
import bench.tritonbench_g.token_attn_llama2.Proofs.Model
import VeriTile.Triton.Float.AddressedInputsIO
import VeriTile.Triton.Float.ProgramEquivalence

/-! The complete score grid's typed channels. Q/K retain arbitrary admitted
strides; metadata retains the checked signed-to-Nat adapter. Out is a masked
strided view, with every unwritten cell protected by the shared exact frame. -/
namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Interface
open VeriTile Triton FP.Structural Whole
noncomputable section

def inputs (c : Call) : List IOChannel :=
  [⟨.fp32,[c.batches,c.heads,c.dim],"Q"⟩,
   ⟨.fp32,[c.cacheRows,c.kvHeads,c.dim],"K"⟩,
   ⟨.nat,[c.batches],"Length"⟩,
   ⟨.nat,[c.batches],"Start"⟩,
   ⟨.nat,[c.locSize],"Loc"⟩]
def outputs (c : Call) : List IOChannel :=
  [⟨.fp32,[c.batches,c.heads,c.maxLength],"Out"⟩]
abbrev Inputs (α : Type) (c : Call) := ChannelValues α (inputs c)
abbrev Outputs (α : Type) (c : Call) := ChannelValues α (outputs c)

def read (c : Call) : (i : Fin (inputs c).length) → TileIndex (inputs c)[i].shape → Nat
  | ⟨0,_⟩ => fun j => j.1.val*c.qBatch+j.2.1.val*c.qHead+j.2.2.1.val*c.qDim
  | ⟨1,_⟩ => fun j => j.1.val*c.kBatch+j.2.1.val*c.kHead+j.2.2.1.val*c.kDim
  | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ => fun j => j.1.val
  | ⟨i+5,h⟩ => False.elim (by simp [inputs] at h)

def query {α : Type} {c : Call} (xs : Inputs α c)
    (b : Fin c.batches) (h : Fin c.heads) (d : Fin c.dim) : α :=
  xs ⟨0,by simp [inputs]⟩ (b,h,d,PUnit.unit)
def key {α : Type} {c : Call} (xs : Inputs α c)
    (row : Fin c.cacheRows) (h : Fin c.kvHeads) (d : Fin c.dim) : α :=
  xs ⟨1,by simp [inputs]⟩ (row,h,d,PUnit.unit)
def lengths {α : Type} {c : Call} (xs : Inputs α c) (b : Fin c.batches) : Nat :=
  xs ⟨2,by simp [inputs]⟩ (b,PUnit.unit)
def starts {α : Type} {c : Call} (xs : Inputs α c) (b : Fin c.batches) : Nat :=
  xs ⟨3,by simp [inputs]⟩ (b,PUnit.unit)
def locations {α : Type} {c : Call} (xs : Inputs α c) (o : Fin c.locSize) : Nat :=
  xs ⟨4,by simp [inputs]⟩ (o,PUnit.unit)

/-- Logical metadata equals the original signed data after its proven exact
nonnegative adapter. This contains no execution or access-safety condition. -/
def MetadataValues {α : Type} (c : Call) (m : Metadata c) (xs : Inputs α c) : Prop :=
  (∀ b, lengths xs b = m.len b.val) ∧
  (∀ b, starts xs b = m.begin b.val) ∧
  (∀ o, locations xs o = (m.locations o).toNat)

def write (c : Call) (m : Metadata c) :
    (o : Fin (outputs c).length) → TileIndex (outputs c)[o].shape → Nat
  | ⟨0,_⟩ => fun j => j.2.1.val*c.outHead+(m.begin j.1.val+j.2.2.1.val)*c.outSeq
  | ⟨i+1,h⟩ => False.elim (by simp [outputs] at h)
def outputMask (c : Call) (m : Metadata c) :
    (o : Fin (outputs c).length) → TileIndex (outputs c)[o].shape → Prop
  | ⟨0,_⟩ => fun j => j.2.2.1.val < m.len j.1.val
  | ⟨i+1,h⟩ => False.elim (by simp [outputs] at h)

def invocation {β : Type} (M : Program → Algebra β) (c : Call) (scale : ℝ)
    (pid : Program) : Invocation β where
  kernel := source (c.parameters scale)
  backend := M pid
  clearRegisters := Bool.true
  pids := ids pid
  numPids := some (dimensions c.grid)

/-- The source is fixed by Kernels. No execution callback or replacement kernel
is accepted. Every invocation receives its actual predecessor memory. -/
def io {α β : Type} (M : Program → Algebra β) (c : Call) (m : Metadata c)
    (scale : ℝ) (work : List Program)
    (encodeAt : RegionName → Nat → α → β) (observe : β → α) : AddressedProgramIO α β where
  inputs := inputs c
  outputs := outputs c
  buffers := ["Q","K","Length","Start","Loc","Out"]
  scratch := []
  read := fun i _ => read c i
  inputMask := fun _ _ _ => True
  write := fun o _ => write c m o
  outputMask := fun o _ => outputMask c m o
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => work.map (invocation M c scale)
  encodeAt := encodeAt
  observe := fun _ _ => observe

/-- Unobserved symbolic source IO on the same logical windows. Identity
encoding permits every symbolic payload, including all addressed native leaves. -/
def symbolicIO {β : Type} (M : Program → Algebra β) (c : Call) (m : Metadata c)
    (scale : ℝ) (work : List Program) : ProgramIO β β where
  toMemoryIO := (io M c m scale work (fun _ _ x => x) id).toMemoryIO
  schedule := fun _ => work.map (invocation M c scale)
  encode := id
  observe := fun _ _ => id

def reuseInvocation {β : Type} (M : Program → Algebra β) (c : Call) (scale : ℝ)
    (pid : Program) : Invocation β :=
  { invocation M c scale pid with kernel := reuseSource (c.parameters scale) }

def reuse {β : Type} (M : Program → Algebra β) (c : Call) (m : Metadata c)
    (scale : ℝ) (work : List Program) : ProgramIO.Variant (symbolicIO M c m scale work) where
  schedule := fun _ => work.map (reuseInvocation M c scale)
  observe := fun _ _ => id

def result {α : Type} {c : Call}
    (f : Fin c.batches → Fin c.heads → Fin c.maxLength → α) : Outputs α c
  | ⟨0,_⟩ => fun j => f j.1 j.2.1 j.2.2.1
  | ⟨i+1,h⟩ => False.elim (by simp [outputs] at h)

end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Interface

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Logical
open VeriTile Triton FP.Structural Whole
noncomputable section

/-- Named tensor and metadata snapshots; no original input coordinate is dropped. -/
structure Inputs (c : Call) where
  query : Fin c.batches → Fin c.heads → Fin c.dim → ℝ
  key : Fin c.cacheRows → Fin c.kvHeads → Fin c.dim → ℝ
  length : Fin c.batches → Nat
  start : Fin c.batches → Nat
  location : Fin c.locSize → Nat

def inputs (c : Call) : Interface.Inputs ℝ c ≃ Inputs c where
  toFun := fun values => ⟨Interface.query values,Interface.key values,
    Interface.lengths values,Interface.starts values,Interface.locations values⟩
  invFun := fun xs port => match port with
    | ⟨0,_⟩ => fun j => xs.query j.1 j.2.1 j.2.2.1
    | ⟨1,_⟩ => fun j => xs.key j.1 j.2.1 j.2.2.1
    | ⟨2,_⟩ => fun j => xs.length j.1
    | ⟨3,_⟩ => fun j => xs.start j.1
    | ⟨4,_⟩ => fun j => xs.location j.1
    | ⟨n+5,h⟩ => False.elim (by simp [Interface.inputs] at h)
  left_inv := by
    intro values
    funext port
    fin_cases port <;> funext j
    all_goals first | (rcases j with ⟨a,b,c,⟨⟩⟩; rfl) | (rcases j with ⟨a,⟨⟩⟩; rfl)
  right_inv := by intro xs; cases xs; rfl

def outputs (c : Call) : Interface.Outputs ℝ c ≃
    (Fin c.batches → Fin c.heads → Fin c.maxLength → ℝ) where
  toFun := fun values b h t => values ⟨0, by simp [Interface.outputs]⟩ (b,h,t,())
  invFun := Interface.result
  left_inv := by
    intro values
    funext port
    fin_cases port
    funext j
    rcases j with ⟨a,b,c,⟨⟩⟩
    rfl
  right_inv := by intro f; rfl

/-- The same total gathered-key lookup as the canonical reference. On active
outputs Valid supplies the row and grouped-head bounds. -/
def Inputs.keyAt {c : Call} (xs : Inputs c) (row head : Nat) (d : Fin c.dim) : ℝ :=
  if hr : row < c.cacheRows ∧ head < c.kvHeads then xs.key ⟨row,hr.1⟩ ⟨head,hr.2⟩ d else 0

def domain (c : Call) (m : Metadata c) (xs : Inputs c) : Prop :=
  (∀ b, xs.length b = m.len b.val) ∧
  (∀ b, xs.start b = m.begin b.val) ∧
  (∀ o, xs.location o = (m.locations o).toNat)

def io (c : Call) (m : Metadata c) (work : List Program) :
    LogicalDomainIO (Interface.Inputs ℝ c) (Interface.Outputs ℝ c)
      (Inputs c) (Fin c.batches → Fin c.heads → Fin c.maxLength → ℝ)
      ((Interface.io (fun _ => realAlgebra) c m (idealScale c.dim) work
        (fun _ _ x => x) id).ImplementsOn) where
  inputs := inputs c
  outputs := outputs c

end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Logical

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Reference
open VeriTile Triton FP.Structural Whole
noncomputable section

def queryAddress (c : Call) (b h d : Nat) : Nat := b*c.qBatch+h*c.qHead+d*c.qDim
def keyAddress (c : Call) (row h d : Nat) : Nat := row*c.kBatch+h*c.kHead+d*c.kDim

/-- A finite logical K lookup, with the original arbitrary key strides. -/
def keyAt {α : Type} [Inhabited α] (c : Call) (xs : Interface.Inputs α c)
    (row h : Nat) (d : Fin c.dim) : α :=
  if hrd : row < c.cacheRows ∧ h < c.kvHeads then
    Interface.key xs ⟨row,hrd.1⟩ ⟨h,hrd.2⟩ d
  else default

end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Reference
