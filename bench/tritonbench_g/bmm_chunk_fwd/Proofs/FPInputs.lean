import bench.tritonbench_g.bmm_chunk_fwd.Proofs.FPGrid

namespace VeriTile.Bench.Repairs.BmmChunkFwd.FPGrid
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section

abbrev Coord := Nat × Nat × Nat × Nat
abbrev SeqCoord := Nat × Nat

def inputOffset (v : Host.Strides) (b r g k : Nat) := b*v.batch+r*v.row+g*v.head+k*v.column

def readInput {α : Type} [Inhabited α] (storage : TypedSource.Storage) (cell : Cell α) : α :=
  match storage with
  | .fp32 => cell.read .real
  | .fp16 => cell.read .fp16
  | .bf16 => cell.read .bf16

/-- Initial raw memory binding of the original direct input buffers. Only input values
and the fresh output region are specified. Bounds and native index facts are
separately derived from admission; no floating result appears as a premise. -/
structure Prepared {α : Type} [Inhabited α] (h : Host.Call) (s : State α) (xs : LogicalInputs α) : Prop where
  freshA : h.A≠h.Out
  freshB : h.B≠h.Out
  freshSeq : Region.cast h.Seq≠h.Out
  aPins : ∀ b r g k, b<h.shape.batch → r<h.shape.seqlen → g<h.shape.groups → k<h.shape.K →
    readInput h.aStorage (s.mem h.A (inputOffset (Host.prepared h.shape h.aStrides) b r g k))=xs.a b r g k
  bPins : ∀ b r g k, b<h.shape.batch → r<h.shape.seqlen → g<h.shape.groups → k<h.shape.K →
    readInput h.bStorage (s.mem h.B (inputOffset (Host.prepared h.shape h.bStrides) b r g k))=xs.b b r g k
  seqPins : h.hasSeq=Bool.true → ∀ b r, b<h.shape.batch → r<h.shape.seqlen →
    (s.mem (Region.cast h.Seq) (b*h.seqBatch+r*h.seqRow)).read .int=xs.seq b r

theorem prepared_stable {α : Type} [Inhabited α] (h : Host.Call) (s t : State α)
    (xs : LogicalInputs α) (prepared : Prepared h s xs) (frame : FPSuccess.RegionFrame h.Out s t) :
    Prepared h t xs := by
  refine {prepared with aPins:=?_,bPins:=?_,seqPins:=?_}
  · intro b r g k hb hr hg hk
    rw [frame h.A _ prepared.freshA]
    exact prepared.aPins b r g k hb hr hg hk
  · intro b r g k hb hr hg hk
    rw [frame h.B _ prepared.freshB]
    exact prepared.bPins b r g k hb hr hg hk
  · intro hs b r hb hr
    rw [frame (Region.cast h.Seq) _ prepared.freshSeq]
    exact prepared.seqPins hs b r hb hr

end
end VeriTile.Bench.Repairs.BmmChunkFwd.FPGrid

#guard_msgs (drop info) in
#auditModuleAxioms
