import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.ForwardComplete
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.HostStages
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.BackwardGrid
import Mathlib.Data.Nat.Log

namespace VeriTile.Bench.Repairs.FastCeLoss.API
open Source Complete Grid HostStages
noncomputable section
set_option maxHeartbeats 4000000

def nextPowerOfTwo (v : Nat) : Nat := 2^(Nat.clog 2 v)
def selectedBlock (s : Shape) : Nat := if s.chunks=1 then nextPowerOfTwo s.vocab else 65536
def selectedWarps (block : Nat) : Nat :=
  if block≥32768 then 32 else if block≥8192 then 16 else if block≥2048 then 8 else 4

structure Setup (s : Shape) (b : Buffers) : Prop where
  positive : 0<s.vocab
  block : s.block=selectedBlock s
  separated : ForwardSeparated b
  singleAlias : s.chunks=1 → b.lse=b.chunks
  chunkFresh : s.chunks≠1 → HostSeparated b

 theorem setup_layout (s : Shape) (b : Buffers) (h : Setup s b) : ForwardLayout s b where
  positive := h.positive
  separated := h.separated
  singleAlias := h.singleAlias
  chunkFresh := h.chunkFresh
  singleCover := by
    intro single
    rw [h.block,selectedBlock,if_pos single]
    exact Nat.le_pow_clog (by decide) _
  chunkBlock := by intro chunked; rw [h.block,selectedBlock,if_neg chunked]

/-- Ordinary fresh scalar/upstream allocations. No value or reduction equation
is present. Upstream strides may be zero (the usual expand view of sum backward). -/
structure PrivateBuffers (b : Buffers) : Prop where
  scalarLogits : b.scalar.storage≠b.logits.storage
  scalarLSE : b.scalar.storage≠b.lse.storage
  scalarLabels : b.scalar.storage≠b.labels.storage
  upstreamLogits : b.upstream.storage≠b.logits.storage
  upstreamLSE : b.upstream.storage≠b.lse.storage
  upstreamLabels : b.upstream.storage≠b.labels.storage

structure Saved where
  logits : Tensor
  lse : Tensor
  labels : Tensor
  options : Options

def saved (o : Options) (b : Buffers) : Saved := ⟨b.logits,b.lse,b.labels,o⟩

def actualForward {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (entry : Memory α) : Memory α :=
  HostStages.finish a s b labels (forwardGrid a o s b labels entry)

theorem actual_forward_eq {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α) (layout : ForwardLayout s b) :
    actualForward a o s b labels entry=forwardSaved a o s b labels entry := by
  by_cases single : s.chunks=1
  · simp only [actualForward,HostStages.finish,forwardSaved,finishGrid,if_pos single]
  · exact HostStages.state_equivalence a s b labels _ (layout.chunkFresh single)

def meanMemory {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (items : Option α) (entry : Memory α) : Memory α :=
  let after := actualForward a o s b labels entry
  writeRows b.scalar 1 (fun _ => scalarLoss a s b labels items after) after

structure ForwardResult (α : Type) where
  memory : Memory α
  returned : Tensor
  customReturned : Tensor
  context : Saved

def forward {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (items : Option α) (entry : Memory α) : ForwardResult α :=
  ⟨meanMemory a o s b labels items entry,b.scalar,b.losses,saved o b⟩

def upstreamWindow (s : Shape) (b : Buffers) (p : Address) : Prop :=
  ∃ row<s.rows, p=addr b.upstream (row*b.upstreamStride)
noncomputable instance (s : Shape) (b : Buffers) : DecidablePred (upstreamWindow s b) :=
  Classical.decPred _

def meanBackward {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (items : Option α) (upstream : α) (m : Memory α) : Memory α :=
  fun p => if upstreamWindow s b p then rowUpstream a s labels items upstream else m p

theorem upstream_here {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (items : Option α) (upstream : α) (m : Memory α)
    (row : Nat) (hr : row<s.rows) :
    meanBackward a s b labels items upstream m (addr b.upstream (row*b.upstreamStride))=
      rowUpstream a s labels items upstream := by
  exact if_pos ⟨row,hr,rfl⟩

theorem upstream_other {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (items : Option α) (upstream : α) (m : Memory α)
    (p : Address) (different : p.1≠b.upstream.storage) :
    meanBackward a s b labels items upstream m p=m p := by
  apply if_neg
  rintro ⟨row,_,h⟩
  exact different (congrArg Prod.fst h)

/-- Saved tensor versions/storage must still match forward. This source call
uses the saved objects directly; no independently supplied saved LSE exists. -/
def backwardEntry {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (items : Option α) (upstream : α) (entry : Memory α) : Memory α :=
  meanBackward a s b labels items upstream (meanMemory a o s b labels items entry)

structure BackwardResult (α : Type) where
  memory : Memory α
  returned : List (Option Tensor)

def backward {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (items : Option α) (upstream : α) (entry : Memory α) : BackwardResult α :=
  ⟨backwardGrid a o s b labels (backwardEntry a o s b labels items upstream entry),backwardTuple b⟩

theorem mean_output {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (items : Option α) (entry : Memory α) (layout : ForwardLayout s b) :
    (forward a o s b labels items entry).memory (addr b.scalar 0)=scalarGraph a o s b labels items entry := by
  change meanMemory a o s b labels items entry _=_
  rw [meanMemory,write_here _ _ _ _ _ (by decide),actual_forward_eq a o s b labels entry layout]
  exact Complete.scalar_loss a o s b labels items entry layout

theorem backward_inputs {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (items : Option α) (upstream : α) (entry : Memory α)
    (layout : ForwardLayout s b) (fresh : PrivateBuffers b) (row col : Nat) (hr : row<s.rows) :
    backwardEntry a o s b labels items upstream entry (addr b.logits (row*s.vocab+col))=
      entry (addr b.logits (row*s.vocab+col)) ∧
    backwardEntry a o s b labels items upstream entry (addr b.lse row)=lseGraph a o s b entry row ∧
    backwardEntry a o s b labels items upstream entry (addr b.upstream (row*b.upstreamStride))=
      rowUpstream a s labels items upstream := by
  refine ⟨?_,?_,upstream_here a s b labels items upstream _ row hr⟩
  · rw [backwardEntry,upstream_other _ _ _ _ _ _ _ _ fresh.upstreamLogits.symm,
      meanMemory,write_other _ _ _ _ _ _ fresh.scalarLogits.symm,
      actual_forward_eq a o s b labels entry layout]
    exact Complete.logits_saved a o s b labels entry layout _
  · rw [backwardEntry,upstream_other _ _ _ _ _ _ _ _ fresh.upstreamLSE.symm,
      meanMemory,write_other _ _ _ _ _ _ fresh.scalarLSE.symm,
      actual_forward_eq a o s b labels entry layout]
    exact Complete.forward_lse a o s b labels entry layout row hr

def gradientGraph {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (items : Option α) (upstream : α) (entry : Memory α) (row col : Nat) : α :=
  backwardScalar a o (labels row) col (entry (addr b.logits (row*s.vocab+col)))
    (lseGraph a o s b entry row) (rowUpstream a s labels items upstream)

theorem gradient_output {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (items : Option α) (upstream : α) (entry : Memory α)
    (layout : ForwardLayout s b) (fresh : PrivateBuffers b) (row col : Nat)
    (hr : row<s.rows) (hc : col<s.vocab) :
    (backward a o s b labels items upstream entry).memory (addr b.logits (row*s.vocab+col))=
      gradientGraph a o s b labels items upstream entry row col := by
  have sep : BackwardSeparated b := by
    refine ⟨?_,fresh.upstreamLogits.symm⟩
    by_cases single : s.chunks=1
    · rw [layout.singleAlias single]
      exact layout.separated.chunksLogits.symm
    · exact (layout.chunkFresh single).lseLogits.symm
  change backwardGrid a o s b labels _ _=_
  rw [Grid.backward_output a o s b labels _ sep layout.positive row col hr hc]
  have h := backward_inputs a o s b labels items upstream entry layout fresh row col hr
  rw [h.1,h.2.1,h.2.2]
  rfl

theorem caller_frame {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (items : Option α) (upstream : α) (entry : Memory α)
    (layout : ForwardLayout s b) (fresh : PrivateBuffers b) (p : Address)
    (chunks : p.1≠b.chunks.storage) (losses : p.1≠b.losses.storage)
    (lse : p.1≠b.lse.storage) (scalar : p.1≠b.scalar.storage) (up : p.1≠b.upstream.storage)
    (outside : p.1≠b.logits.storage ∨ p.2<b.logits.offset ∨ b.logits.offset+s.rows*s.vocab≤p.2) :
    (backward a o s b labels items upstream entry).memory p=entry p := by
  have sep : BackwardSeparated b := by
    refine ⟨?_,fresh.upstreamLogits.symm⟩
    by_cases single : s.chunks=1
    · rw [layout.singleAlias single]
      exact layout.separated.chunksLogits.symm
    · exact (layout.chunkFresh single).lseLogits.symm
  change backwardGrid a o s b labels _ p=_
  rw [Grid.backward_frame a o s b labels _ sep layout.positive p outside,
    backwardEntry,upstream_other _ _ _ _ _ _ _ _ up]
  unfold meanMemory
  have notScalar : ¬window b.scalar 1 p := by intro h; exact scalar h.1
  rw [writeRows,if_neg notScalar,actual_forward_eq a o s b labels entry layout]
  exact Complete.forward_frame a o s b labels entry layout p chunks losses lse

theorem exact_returns {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (items : Option α) (upstream : α) (entry : Memory α) :
    (forward a o s b labels items entry).returned=b.scalar ∧
    (forward a o s b labels items entry).customReturned=b.losses ∧
    (forward a o s b labels items entry).context=saved o b ∧
    (backward a o s b labels items upstream entry).returned=[some b.logits,none,none,none] := by
  exact ⟨rfl,rfl,rfl,rfl⟩

#axiomsClean setup_layout
#axiomsClean mean_output
#axiomsClean gradient_output
#axiomsClean caller_frame
#axiomsClean exact_returns
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.API
