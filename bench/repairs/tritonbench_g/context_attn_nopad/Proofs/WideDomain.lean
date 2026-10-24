import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.NativeProgram

/-! The ordinary prepared tensor domain without the historical int32 metadata
premises. A nonempty launch supplies an output head and channel. Injectivity
then forces positive row stride when multiple rows exist, so the row count is
bounded by the finite output allocation. This derives native control bounds
without imposing a selected safe shape or declaring large metadata a failure.
-/
namespace VeriTile.Bench.Repairs.ContextAttnNopad.WideDomain
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section

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

def Shape.ofLegacy {c : Call} (h : Host.Shape c) : Shape c :=
  ⟨h.width,h.kDims,h.vDims,h.oDims,h.kHeads,h.vHeads,h.oHeads,
    h.qRows,h.kRows,h.vRows,h.oRows,h.intervals,h.outputInjective⟩

def Memory.ofLegacy {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (h : Host.Memory c r A s) : Memory c r A s :=
  ⟨Shape.ofLegacy h.shape,h.start,h.length,h.startExtent,h.lengthExtent,
    h.qBounded,h.kBounded,h.vBounded,h.oBounded,h.outQuery,h.outKey,h.outValue,h.outStart,h.outSequence⟩

theorem dims_positive {c : Call} (hs : Shape c) : 0 < c.q.dims := by
  rcases hs.width with h | h | h | h <;> omega

/-- Finite injective output storage bounds the logical row count, including
zero/one rows and noncontiguous output strides. No metadata width is used. -/
theorem output_rows_le_extent {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) (hh : 0 < c.q.heads) : c.o.rows ≤ A.extent r.out := by
  have hoh : 0 < c.o.heads := lt_of_lt_of_le hh hm.shape.oHeads
  have hod : 0 < c.o.dims := lt_of_lt_of_le (dims_positive hm.shape) hm.shape.oDims
  by_cases hzero : c.o.rows = 0
  · omega
  have hr : 0 < c.o.rows := by omega
  by_cases hone : c.o.rows = 1
  · have h := hm.oBounded 0 0 0 hr hoh hod
    simp only [Tensor3.offset,Nat.zero_mul,Nat.zero_add] at h
    omega
  have htwo : 1 < c.o.rows := by omega
  have stride : 0 < c.o.rowStride := by
    by_contra hn
    have hz : c.o.rowStride = 0 := by omega
    have he : c.o.offset 0 0 0 = c.o.offset 1 0 0 := by simp [Tensor3.offset,hz]
    have := (hm.shape.outputInjective 0 1 0 0 0 0 hr htwo hoh hoh hod hod he).1
    omega
  have hb := hm.oBounded (c.o.rows-1) 0 0 (by omega) hoh hod
  simp only [Tensor3.offset,Nat.zero_mul,Nat.add_zero] at hb
  have hm' : c.o.rows-1 ≤ (c.o.rows-1)*c.o.rowStride := by
    exact Nat.le_mul_of_pos_right _ stride
  omega

theorem output_rows_bound {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) (ha : NativeProgram.ABI c A) (hh : 0 < c.q.heads) :
    c.o.rows ≤ 2^61 := by
  have hr := output_rows_le_extent hm hh
  have hb := ha.bytes r.out
  norm_num at hb ⊢
  omega

theorem metadata_bound {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) (ha : NativeProgram.ABI c A) (hh : 0 < c.q.heads)
    (b : Nat) (hb : b < c.batches) : c.start b + c.length b ≤ 2^61 :=
  le_trans (hm.shape.oRows b hb) (output_rows_bound hm ha hh)

theorem maximum_bound {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) (ha : NativeProgram.ABI c A) (hh : 0 < c.q.heads) :
    HostPlan.maximum c ≤ 2^61 := by
  apply Finset.sup_le
  intro b _
  have := metadata_bound hm ha hh b.val b.isLt
  omega

theorem metadataMaximum_eq {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) : HostPlan.metadataMaximum c r s = HostPlan.maximum c := by
  unfold HostPlan.metadataMaximum HostPlan.maximum
  congr 1
  funext b
  rw [hm.length b b.isLt,Int.toNat_natCast]

theorem effective_blocks_bound {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) (ha : NativeProgram.ABI c A) (hh : 0 < c.q.heads) :
    (HostPlan.effective c r s).blocks ≤ 2^54 := by
  have h := maximum_bound hm ha hh
  rw [← metadataMaximum_eq hm] at h
  change (min c.maxInputLen (HostPlan.metadataMaximum c r s)+127)/128 ≤ _
  norm_num at h ⊢
  omega

/-- Every rounded end and the last range backedge are signed64-exact. -/
theorem loop_end_bound {blocks block : Nat} (hb : blocks ≤ 2^54) (hi : block < blocks) :
    (block+1)*128 ≤ 2^61 ∧ (block+1)*128 < 2^63 := by
  norm_num at hb ⊢
  omega

theorem key_lane_and_backedge {blocks block iter lane : Nat}
    (hb : blocks ≤ 2^54) (hi : block < blocks) (hit : iter ≤ block) (hl : lane < 128) :
    iter*128+lane < 2^63 ∧ iter*128+128 < 2^63 := by
  have := loop_end_bound hb hi
  omega

theorem no_jobs_without_heads (c : Call) (hh : c.q.heads = 0) :
    IsEmpty (HostPlan.Job c) := by
  constructor
  intro j
  have := (HostPlan.global j.2.1).isLt
  omega

#print axioms output_rows_bound
#print axioms effective_blocks_bound
end
end VeriTile.Bench.Repairs.ContextAttnNopad.WideDomain
