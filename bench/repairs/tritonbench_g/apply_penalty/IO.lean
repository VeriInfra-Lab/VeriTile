import VeriTile.Triton.Memory.LogicalIO
import VeriTile.Triton.Memory.KernelSpec.Scatter
import bench.repairs.tritonbench_g.apply_penalty.Kernels
import bench.repairs.tritonbench_g.apply_penalty.Proofs.Rewrite
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.ApplyPenalty.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Three FP32 scalar penalties, two cumsum scalar slots, active IDs/counts,
and the gathered FP32 input logits. Integer cells retain the selected source's
nonnegative signed-int32 projection. -/
@[reducible] def inputs (B : Nat) : List IOChannel :=
  [⟨.real,[],"presence"⟩,⟨.real,[],"frequency"⟩,⟨.real,[],"repetition"⟩,
   ⟨.nat,[],"cumsum"⟩,⟨.nat,[],"cumsum"⟩,
   ⟨.nat,[B],"ids"⟩,⟨.nat,[B],"counts"⟩,⟨.real,[B],"logits"⟩]

abbrev Inputs (α : Type) (B : Nat) := ChannelValues α (inputs B)

def active {α : Type} {B : Nat} (xs : Inputs α B) (j : TileIndex [B]) : Prop :=
  Nat.lt (Nat.add (xs 3 PUnit.unit) j.1.val) (xs 4 PUnit.unit)

def address {α : Type} {B : Nat} (stride pid : Nat) (xs : Inputs α B)
    (j : TileIndex [B]) : Nat := Nat.add (pid*stride) (xs 5 j)

def launch {α : Type} (M : Algebra α) (candidate : Bool) (stride B pid : Nat) :
    Invocation α where
  kernel := if candidate then Rewrite.reciprocalSource stride B else Rewrite.originalSource stride B
  backend := FP.ProfiledRewrite.engine M
  pids := fun ax => if ax=0 then pid else 0

/-- The actual FP32 source has in-place gather/scatter, metadata-dependent
windows and masks, arbitrary program id/stride/block, and no private scratch.
No injectivity condition is imposed on floating equivalence: both sources
retain precisely the same store order even when token IDs repeat. -/
@[reducible] def io {α : Type} (M : Algebra α) (stride B pid : Nat) : ProgramIO α α where
  inputs := inputs B
  outputs := [⟨.real,[B],"logits"⟩]
  scratch := []
  buffers := ["presence","frequency","repetition","cumsum","ids","counts","logits"]
  read := fun i xs => match i with
    | ⟨0,_⟩ => fun _ => pid
    | ⟨1,_⟩ => fun _ => pid
    | ⟨2,_⟩ => fun _ => pid
    | ⟨3,_⟩ => fun _ => pid
    | ⟨4,_⟩ => fun _ => pid+1
    | ⟨5,_⟩ => fun j => Nat.add (xs 3 PUnit.unit) j.1.val
    | ⟨6,_⟩ => fun j => Nat.add (xs 3 PUnit.unit) j.1.val
    | ⟨7,_⟩ => fun j => address stride pid xs j
  inputMask := fun i xs => match i with
    | ⟨0,_⟩ => fun _ => True
    | ⟨1,_⟩ => fun _ => True
    | ⟨2,_⟩ => fun _ => True
    | ⟨3,_⟩ => fun _ => True
    | ⟨4,_⟩ => fun _ => True
    | ⟨5,_⟩ => fun j => active xs j
    | ⟨6,_⟩ => fun j => active xs j
    | ⟨7,_⟩ => fun j => active xs j
  write := fun i xs => match i with | ⟨0,_⟩ => fun j => address stride pid xs j
  outputMask := fun i xs => match i with | ⟨0,_⟩ => fun j => active xs j
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => [launch M Bool.false stride B pid]
  encode := id
  observe := fun _ _ => id

def candidate {α : Type} (M : Algebra α) (stride B pid : Nat) :
    (io M stride B pid).Variant where
  schedule := fun _ => [launch M Bool.true stride B pid]
  observe := fun _ _ => id

/-- Exactly the existing DIV-MUL-RCP domain, stated on logical FP32 inputs.
Only active lanes with a true positive comparison require finite loaded
operands and nonzero loaded repetition. Count conversion, unordered comparisons,
inactive padding and the unchanged product arm gain no new conditions. -/
def DivisionDomain {α : Type} (M : Algebra α) (D : FP.Guarded.Domain α)
    (lt : α→α→Bool) {B : Nat} (xs : Inputs α B) : Prop :=
  ∀ j, active xs j →
    lt (M.literal (some .fp32) .real 0) (M.fp32Load (xs 7 j)) = Bool.true →
      D .finite (M.fp32Load (xs 7 j)) ∧
      D .finite (M.fp32Load (xs 2 PUnit.unit)) ∧
      D .nonzero (M.fp32Load (xs 2 PUnit.unit))

end
end VeriTile.Bench.Repairs.ApplyPenalty.PublicIO

namespace VeriTile.Bench.Repairs.ApplyPenalty.PublicIO
open VeriTile.Triton

/-- Scalar penalties at pid; cumsum bounds at pid and pid+1; ids and counts at
start+j. Logits are read and overwritten at pid * stride_logit_b + ids[j], with
the same tail mask start+j < stop. The IO contract checks injectivity only on
active write lanes, so padding does not need distinct token ids. -/
def applyPenaltyIO
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    MetaScatterMasked2DKernelIO₁ where
  kernel := kernel Logits presence_penalty freqency_penalty
    repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
    stride_logit_b stride_logit_s BLOCK_P
  projection := by rfl
  fbuf1 := presence_penalty
  fbuf2 := freqency_penalty
  fbuf3 := repetition_penalty
  mbuf := p_cumsum_seq_len
  idbuf := p_token_ids
  cntbuf := p_token_counts
  inp := Logits
  out := Logits
  B := BLOCK_P
  fwin1 := fun pid₀ _ => pid₀
  fwin2 := fun pid₀ _ => pid₀
  fwin3 := fun pid₀ _ => pid₀
  mwin1 := fun pid₀ _ => pid₀
  mwin2 := fun pid₀ _ => pid₀ + 1
  readi := fun _ _ m₁ _ j => m₁ + j.val
  readc := fun _ _ m₁ _ j => m₁ + j.val
  read := fun pid₀ _ _ _ ids j => pid₀ * stride_logit_b + ids j
  mask := fun _ _ m₁ m₂ j => m₁ + j.val < m₂
  write := fun pid₀ _ _ _ ids j => pid₀ * stride_logit_b + ids j

/-- Identical metadata slots, masks, in-place gather/scatter and active-index requirement. -/
def reciprocalIO (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) : MetaScatterMasked2DKernelIO₁ :=
  { applyPenaltyIO Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P with
    kernel := reciprocalKernel Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
    projection := by rfl }


end VeriTile.Bench.Repairs.ApplyPenalty.PublicIO

namespace VeriTile.Bench.Repairs.ApplyPenalty.LogicalIO
open VeriTile Triton PublicIO
noncomputable section

/-- Loaded logical penalties and token tiles. Program coordinates, sequence
bounds and token IDs remain present for the exact metadata-dependent masks. -/
structure Inputs (B : Nat) where
  program : Nat
  secondaryProgram : Nat
  presence : ℝ
  frequency : ℝ
  repetition : ℝ
  sequenceStart : Nat
  sequenceEnd : Nat
  tokenIds : Fin B → Nat
  counts : Fin B → Nat
  logits : Fin B → ℝ

/-- Only curry the formula arguments; keep the full existing scatter judgment. -/
def judgment (core : MetaScatterMasked2DKernelIO₁)
    (f : Inputs core.B → Fin core.B → ℝ) : Prop :=
  core.Implements (fun p q presence frequency repetition start stop ids counts logits =>
    f ⟨p,q,presence,frequency,repetition,start,stop,ids,counts,logits⟩)

def io (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    LogicalIO (Inputs BLOCK_P) (Fin BLOCK_P → ℝ) (Inputs BLOCK_P) (Fin BLOCK_P → ℝ)
      (judgment (applyPenaltyIO Logits presence_penalty freqency_penalty repetition_penalty
      p_token_ids p_token_counts p_cumsum_seq_len stride_logit_b stride_logit_s BLOCK_P)) where
  inputs := Equiv.refl _
  outputs := Equiv.refl _

end
end VeriTile.Bench.Repairs.ApplyPenalty.LogicalIO
