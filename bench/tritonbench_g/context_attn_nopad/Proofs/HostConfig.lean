import bench.tritonbench_g.context_attn_nopad.Proofs.Program
import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.Launch.Composition

/-! The public Python wrapper's FP32 raw-pointer host boundary. Tensor offsets
are relative to the passed data pointer, so independent nonnegative strides are retained.
Metadata is the contiguous int32 ABI actually loaded by the source. The wrapper
allocates no result: it mutates `o` and returns Python `None`.
This boundary describes physical storage, not Torch lazy-negative view flags.
`LogicalViewDefect` proves that the original wrapper's unrestricted logical
tensor interface is false: a negative V view is passed without materialization. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
open VeriTile.Triton Formula Program
noncomputable section

structure Tensor3 where
  rows : Nat
  heads : Nat
  dims : Nat
  rowStride : Nat
  headStride : Nat
  dimStride : Nat

def Tensor3.offset (t : Tensor3) (row head dim : Nat) : Nat :=
  row*t.rowStride + head*t.headStride + dim*t.dimStride

def Tensor3.Bounded (t : Tensor3) (extent : Nat) : Prop :=
  ∀ row head dim, row < t.rows → head < t.heads → dim < t.dims →
    t.offset row head dim < extent

structure Call where
  q : Tensor3
  k : Tensor3
  v : Tensor3
  o : Tensor3
  batches : Nat
  maxInputLen : Nat
  start : Nat → Nat
  length : Nat → Nat

def Call.layout (c : Call) : Layout where
  BM := 128
  BN := 128
  D := c.q.dims
  qRow := c.q.rowStride
  qHead := c.q.headStride
  qDim := c.q.dimStride
  kRow := c.k.rowStride
  kHead := c.k.headStride
  kDim := c.k.dimStride
  vRow := c.v.rowStride
  vHead := c.v.headStride
  vDim := c.v.dimStride
  outRow := c.o.rowStride
  outHead := c.o.headStride
  outDim := c.o.dimStride

def Call.blocks (c : Call) : Nat := (c.maxInputLen + 127) / 128

def Call.grid (c : Call) : Grid := ⟨[c.batches, c.q.heads, c.blocks]⟩

/-- Exact binary64 results of Python `1.0 / (Lq**0.5)` for the four admitted
widths. The subsequent fp32 scalar conversion stays in the typed source.
These are not identified with an exact real reciprocal square root. -/
def Call.scale (c : Call) : ℝ :=
  if c.q.dims = 16 then 1/4 else
  if c.q.dims = 32 then 1592262918131443/9007199254740992 else
  if c.q.dims = 64 then 1/8 else 1592262918131443/18014398509481984

def Call.idealScale (c : Call) : ℝ := 1 / Real.sqrt c.q.dims
def Call.numWarps (c : Call) : Nat := if c.q.dims ≤ 64 then 4 else 8
def Call.source (c : Call) (r : Regions) (swapped : Bool) : ComputeKernel :=
  kernel c.layout r .fp32 swapped c.scale

def Call.launch (c : Call) (idx : GridIndex c.grid) : Launch :=
  ⟨(idx ⟨2, by simp [Call.grid, Grid.rank]⟩).val,
   (idx ⟨0, by simp [Call.grid, Grid.rank]⟩).val,
   (idx ⟨1, by simp [Call.grid, Grid.rank]⟩).val⟩

theorem launch_grid (c : Call) (s : BlockState) (idx : GridIndex c.grid) :
    Program.launch (s.withGridIndex idx) = c.launch idx := by
  simp [Program.launch, Call.launch, BlockState.withGridIndex, BlockState.withPids,
    BlockState.withNumPids, GridIndex.toPids, Call.grid, Grid.rank]

theorem inputs_grid (c : Call) (s : BlockState) (r : Regions) (idx : GridIndex c.grid) :
    inputs (s.withGridIndex idx) r = inputs s r := rfl

theorem launch_batch_lt (c : Call) (idx : GridIndex c.grid) :
    (c.launch idx).batch < c.batches := (idx ⟨0, by simp [Call.grid, Grid.rank]⟩).isLt

theorem launch_head_lt (c : Call) (idx : GridIndex c.grid) :
    (c.launch idx).head < c.q.heads := (idx ⟨1, by simp [Call.grid, Grid.rank]⟩).isLt

theorem launch_block_lt (c : Call) (idx : GridIndex c.grid) :
    (c.launch idx).block < c.blocks := (idx ⟨2, by simp [Call.grid, Grid.rank]⟩).isLt

/-- Conditions on tensor views and metadata, independent of execution/output.
The row interval condition permits arbitrary disjoint packed intervals and
gaps. It does not require contiguous Q/K/V/O tensor strides. -/
structure Shape (c : Call) : Prop where
  width : c.q.dims = 16 ∨ c.q.dims = 32 ∨ c.q.dims = 64 ∨ c.q.dims = 128
  kDims : c.k.dims = c.q.dims
  vDims : c.v.dims = c.q.dims
  oDims : c.q.dims ≤ c.o.dims
  kHeads : c.q.heads ≤ c.k.heads
  vHeads : c.q.heads ≤ c.v.heads
  oHeads : c.q.heads ≤ c.o.heads
  qRows : ∀ b, b < c.batches → c.start b + c.length b ≤ c.q.rows
  kRows : ∀ b, b < c.batches → c.start b + c.length b ≤ c.k.rows
  vRows : ∀ b, b < c.batches → c.start b + c.length b ≤ c.v.rows
  oRows : ∀ b, b < c.batches → c.start b + c.length b ≤ c.o.rows
  intervals : ∀ b b' t t', b < c.batches → b' < c.batches →
    t < c.length b → t' < c.length b' → c.start b+t = c.start b'+t' → b = b' ∧ t = t'
  outputInjective : ∀ row row' head head' dim dim',
    row < c.o.rows → row' < c.o.rows → head < c.o.heads → head' < c.o.heads →
    dim < c.o.dims → dim' < c.o.dims →
    c.o.offset row head dim = c.o.offset row' head' dim' →
    row = row' ∧ head = head' ∧ dim = dim'
  int32Start : ∀ b, b < c.batches → c.start b < 2^31
  int32Length : ∀ b, b < c.batches → c.length b < 2^31

structure Memory (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState) : Prop where
  shape : Shape c
  start : ∀ b, b < c.batches → (inputs s r).start b = (c.start b : Int)
  length : ∀ b, b < c.batches → (inputs s r).sequence b = (c.length b : Int)
  startExtent : c.batches ≤ A.extent r.start
  lengthExtent : c.batches ≤ A.extent r.sequence
  qBounded : c.q.Bounded (A.extent r.query)
  kBounded : c.k.Bounded (A.extent r.key)
  vBounded : c.v.Bounded (A.extent r.value)
  oBounded : c.o.Bounded (A.extent r.out)
  outQuery : r.out ≠ r.query
  outKey : r.out ≠ r.key
  outValue : r.out ≠ r.value
  outStart : r.out ≠ r.start
  outSequence : r.out ≠ r.sequence

def Call.completeGrid (c : Call) : Prop := ∀ b, b < c.batches → c.length b ≤ c.maxInputLen

/-- Causal attention over the initial physical arrays and their own strides.
No program ID, block count, padded key or output occurs in this target. -/
def Call.score (c : Call) (a : Inputs) (b h t j : Nat) : ℝ :=
  (∑ d : Fin c.q.dims,
    a.q (c.q.offset (c.start b+t) h d.val) *
    a.k (c.k.offset (c.start b+j) h d.val)) * c.scale

def Call.expected (c : Call) (a : Inputs) (b h t d : Nat) : ℝ :=
  (∑ j ∈ Finset.range (t+1), Real.exp (c.score a b h t j) *
    a.v (c.v.offset (c.start b+j) h d)) /
  ∑ j ∈ Finset.range (t+1), Real.exp (c.score a b h t j)

def Call.output (c : Call) (b h t d : Nat) : Nat := c.o.offset (c.start b+t) h d

/-- Python's bare return has no returned tensor; `o` is the observable result. -/
def pythonReturn : Option Unit := none

end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
