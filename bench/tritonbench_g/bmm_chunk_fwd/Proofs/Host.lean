import bench.tritonbench_g.bmm_chunk_fwd.Source
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Mathematical_RealViews
import VeriTile.Triton.Launch.Composition

/-! Host argument construction for `_bmm_chunk_fwd`. Tensor copies are an
external value-preserving interface, not an implementation of torch.contiguous.
All strides below are element strides; output allocation is the actual dense
4D/5D shape. The default precision is FP32, with both narrow source choices
retained. No condition initializes the cells of torch.empty. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.Host
open VeriTile.Triton
noncomputable section

def Call.parameters (h : Call) : RealProgram.Parameters where
  A := h.A
  B := h.B
  Out := h.Out
  Seq := h.Seq
  seqlen := h.shape.seqlen
  chunk := h.shape.chunk
  K := h.shape.K
  groups := h.shape.groups
  aBatch := (prepared h.shape h.aStrides).batch
  aRow := (prepared h.shape h.aStrides).row
  aHead := (prepared h.shape h.aStrides).head
  aColumn := (prepared h.shape h.aStrides).column
  bBatch := (prepared h.shape h.bStrides).batch
  bRow := (prepared h.shape h.bStrides).row
  bHead := (prepared h.shape h.bStrides).head
  bColumn := (prepared h.shape h.bStrides).column
  oBatch := (h.shape.chunks * h.shape.groups) * h.shape.chunk * h.shape.chunk
  oChunk := h.shape.groups * h.shape.chunk * h.shape.chunk
  oHead := if h.shape.grouped then h.shape.chunk * h.shape.chunk else 0
  oRow := h.shape.chunk
  oColumn := 1
  sBatch := if h.hasSeq then h.seqBatch else 0
  sRow := if h.hasSeq then h.seqRow else 0
  causal := h.causal
  hasSeq := h.hasSeq

/-- External tensors after the host's optional contiguous copies. Bounds and
value preservation are about inputs only; no executed result is a premise. -/
structure Inputs (h : Call) where
  a : Nat → Nat → Nat → Fin h.shape.K → ℝ
  b : Nat → Nat → Nat → Fin h.shape.K → ℝ
  seq : Nat → Nat → Int

def tileInputs (h : Call) (c : Config) (x : Inputs h) (p₀ p₁ p₂ : Nat) :
    RealInputs.Inputs c.M c.N h.parameters.K where
  a := fun i k => x.a p₁ (p₂ / h.shape.groups * h.shape.chunk +
    RealViews.row c.M c.N h.parameters p₀ i) (p₂ % h.shape.groups) k
  b := fun k j => x.b p₁ (p₂ / h.shape.groups * h.shape.chunk +
    RealViews.column c.N h.parameters p₀ j) (p₂ % h.shape.groups) k
  rowIDs := fun i => x.seq p₁ (p₂ / h.shape.groups * h.shape.chunk +
    RealViews.row c.M c.N h.parameters p₀ i)
  columnIDs := fun j => x.seq p₁ (p₂ / h.shape.groups * h.shape.chunk +
    RealViews.column c.N h.parameters p₀ j)

/-- Independent mathematical target, indexed by the returned tensor's logical
coordinates. Only active tiles are required, exactly as in the source. -/
def expected (h : Call) (x : Inputs h) (batch chunk head row col : Nat) : ℝ :=
  if row < min h.shape.chunk (h.shape.seqlen - chunk * h.shape.chunk) ∧
      col < min h.shape.chunk (h.shape.seqlen - chunk * h.shape.chunk) ∧
      (h.hasSeq = Bool.false ∨ x.seq batch (chunk * h.shape.chunk + row) =
        x.seq batch (chunk * h.shape.chunk + col))
  then ∑ k : Fin h.shape.K,
    x.a batch (chunk * h.shape.chunk + row) head k *
    x.b batch (chunk * h.shape.chunk + col) head k
  else 0

theorem expected_tile (h : Call) (c : Config) (x : Inputs h)
    (p₀ p₁ p₂ : Nat) (i : TileIndex [c.M, c.N]) :
    RealViews.expected c.M c.N h.parameters p₀ p₁ p₂ (tileInputs h c x p₀ p₁ p₂) i =
    expected h x p₁ (p₂ / h.shape.groups) (p₂ % h.shape.groups)
      (RealViews.row c.M c.N h.parameters p₀ i.1)
      (RealViews.column c.N h.parameters p₀ i.2.1) := rfl

end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.Host

#guard_msgs (drop info) in
#auditModuleAxioms
