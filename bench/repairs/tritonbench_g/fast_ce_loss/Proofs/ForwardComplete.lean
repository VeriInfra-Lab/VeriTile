import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.HostGrid
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Geometry

namespace VeriTile.Bench.Repairs.FastCeLoss.Complete
open Source Grid
noncomputable section
set_option maxHeartbeats 3000000

/-- Source host setup: fresh work allocations, actual single-branch alias,
and the source-selected block covering the logical vocabulary. -/
structure ForwardLayout (s : Shape) (b : Buffers) : Prop where
  positive : 0<s.vocab
  separated : ForwardSeparated b
  singleAlias : s.chunks=1 → b.lse=b.chunks
  chunkFresh : s.chunks≠1 → HostSeparated b
  singleCover : s.chunks=1 → s.vocab≤s.block
  chunkBlock : s.chunks≠1 → s.block=65536

def forwardSaved {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α) : Memory α :=
  finishGrid a s b labels (forwardGrid a o s b labels entry)

def lseGraph {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (entry : Memory α) (row : Nat) : α :=
  if s.chunks=1 then forwardLSE a o s b entry (row,0)
  else a.hostLSE (fun c : Fin s.chunks => a.load (forwardLSE a o s b entry (row,c.val)))

def lossGraph {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α) (row : Nat) : α :=
  if s.chunks=1 then forwardLoss a o s b labels entry (row,0)
  else if labels row = -100 then a.literal 0
  else a.hostAdd (a.load (forwardLoss a o s b labels entry (row,0)))
    (a.load (lseGraph a o s b entry row))

def scalarGraph {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (items : Option α) (entry : Memory α) : α :=
  a.hostDiv (a.load (a.hostSum (fun row : Fin s.rows => a.load (lossGraph a o s b labels entry row.val))))
    (denominator a s labels items)

theorem global_lse_graph {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α) (layout : ForwardLayout s b)
    (chunked : s.chunks≠1) (row : Nat) (hr : row<s.rows) :
    globalLSE a s b (forwardGrid a o s b labels entry) row=lseGraph a o s b entry row := by
  simp only [globalLSE,lseGraph,if_neg chunked]
  congr 1
  funext chunk
  rw [Grid.forward_lse a o s b labels entry layout.separated row chunk.val hr chunk.isLt]

theorem forward_lse {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α) (layout : ForwardLayout s b)
    (row : Nat) (hr : row<s.rows) :
    forwardSaved a o s b labels entry (addr b.lse row)=lseGraph a o s b entry row := by
  by_cases single : s.chunks=1
  · have sameTensor := layout.singleAlias single
    simp only [forwardSaved,finishGrid,if_pos single,lseGraph]
    rw [sameTensor]
    have h := Grid.forward_lse a o s b labels entry layout.separated row 0 hr (by omega)
    simpa only [single,Nat.mul_one,Nat.add_zero] using h
  · rw [forwardSaved,finish_output a s b labels _ (layout.chunkFresh single) single row hr _ (Or.inl rfl)]
    simp only [finishValues,if_pos rfl]
    exact global_lse_graph a o s b labels entry layout single row hr

theorem forward_loss {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α) (layout : ForwardLayout s b)
    (row : Nat) (hr : row<s.rows) :
    forwardSaved a o s b labels entry (addr b.losses row)=lossGraph a o s b labels entry row := by
  have positive : 0<s.chunks := Geometry.ceil_positive _ _ (by decide) layout.positive
  by_cases single : s.chunks=1
  · simp only [forwardSaved,finishGrid,if_pos single,lossGraph]
    exact Grid.forward_loss a o s b labels entry layout.separated row hr positive
  · rw [forwardSaved,finish_output a s b labels _ (layout.chunkFresh single) single row hr _ (Or.inr rfl)]
    have different : addr b.losses row≠addr b.lse row := by
      intro h
      exact (layout.chunkFresh single).lseLosses (congrArg Prod.fst h.symm)
    simp only [finishValues,if_neg different,if_neg single,lossGraph]
    rw [Grid.forward_loss a o s b labels entry layout.separated row hr positive,
      global_lse_graph a o s b labels entry layout single row hr]

theorem scalar_loss {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (items : Option α) (entry : Memory α)
    (layout : ForwardLayout s b) :
    scalarLoss a s b labels items (forwardSaved a o s b labels entry)=
      scalarGraph a o s b labels items entry := by
  unfold scalarLoss scalarGraph
  congr 3
  funext row
  rw [forward_loss a o s b labels entry layout row.val row.isLt]

theorem forward_frame {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α) (layout : ForwardLayout s b)
    (point : Address) (chunkOutside : point.1≠b.chunks.storage)
    (lossOutside : point.1≠b.losses.storage) (lseOutside : point.1≠b.lse.storage) :
    forwardSaved a o s b labels entry point=entry point := by
  unfold forwardSaved
  have first : forwardGrid a o s b labels entry point=entry point := by
    apply Grid.forward_frame a o s b labels entry layout.separated
    intro q _ writes
    rcases writes with h|⟨_,h⟩
    · exact chunkOutside (congrArg Prod.fst h)
    · exact lossOutside (congrArg Prod.fst h)
  by_cases single : s.chunks=1
  · simpa only [finishGrid,if_pos single] using first
  · rw [finish_frame a s b labels _ (layout.chunkFresh single)]
    · exact first
    · intro row _ writes
      rcases writes with h|h
      · exact lseOutside (congrArg Prod.fst h)
      · exact lossOutside (congrArg Prod.fst h)

theorem logits_saved {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α) (layout : ForwardLayout s b)
    (i : Nat) : forwardSaved a o s b labels entry (addr b.logits i)=entry (addr b.logits i) := by
  apply forward_frame a o s b labels entry layout
  · exact layout.separated.chunksLogits.symm
  · exact layout.separated.lossesLogits.symm
  · by_cases single : s.chunks=1
    · rw [layout.singleAlias single]
      exact layout.separated.chunksLogits.symm
    · exact (layout.chunkFresh single).lseLogits.symm

#axiomsClean forward_lse
#axiomsClean forward_loss
#axiomsClean scalar_loss
#axiomsClean forward_frame
#axiomsClean logits_saved
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Complete
