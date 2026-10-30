import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.HostParameters
import bench.tritonbench_g.context_attn_bloom.Proofs.RealIO
import bench.tritonbench_g.context_attn_fwd.Proofs.HostComposition

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Host
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
noncomputable section
set_option maxHeartbeats 1000000

/-- Ordinary packed-context tensor interface. Metadata vectors have raw,
unit-stride signed-int32 ABI; arbitrary metadata views are not silently
reinterpreted. Bounds below are only the actual reached source addresses. -/
structure Admissible (a : Arguments) (r : Regions) (A : FlatAlloc) (s : BlockState) : Prop where
  heads_positive : 0 < a.q.heads
  kv_positive : 0 < a.k.heads
  grouped : a.k.heads ∣ a.q.heads
  qk_dims : a.q.dims = a.k.dims
  kv_dims : a.k.dims = a.v.dims
  out_dims : a.o.dims = a.k.dims
  out_heads : a.o.heads = a.q.heads
  value_heads : a.v.heads = a.k.heads
  head_dim : 0 < a.q.dims
  metadata : ∀ b, b < a.batch →
    0 ≤ (inputs s r).start b ∧ 0 ≤ (inputs s r).prompt b ∧
    (inputs s r).prompt b ≤ (inputs s r).sequence b ∧
    (inputs s r).sequence b - (inputs s r).prompt b ≤ a.maxInput ∧
    (inputs s r).start b + ((inputs s r).sequence b - (inputs s r).prompt b) ≤ a.q.rows ∧
    (inputs s r).start b + ((inputs s r).sequence b - (inputs s r).prompt b) ≤ a.o.rows ∧
    (inputs s r).sequence b ≤ a.requestColumns ∧
    0 ≤ (inputs s r).request b ∧ (inputs s r).request b < a.requestRows
  packed : ∀ b c, b < a.batch → c < a.batch → ∀ i j : Int,
    0 ≤ i → i < (inputs s r).sequence b - (inputs s r).prompt b →
    0 ≤ j → j < (inputs s r).sequence c - (inputs s r).prompt c →
    (inputs s r).start b+i = (inputs s r).start c+j → b = c ∧ i = j
  output_layout : a.o.Injective
  separate : r.out ≠ r.query ∧ r.out ≠ r.key ∧ r.out ≠ r.value ∧
    r.out ≠ r.start ∧ r.out ≠ r.sequence ∧ r.out ≠ r.prompt ∧
    r.out ≠ r.request ∧ r.out ≠ r.tokens
  bounds : ∀ idx : GridIndex (grid a),
    AddressBounds.Valid (layout a) r (launchOf a idx) (inputs s r) A.extent

theorem head_lt (a : Arguments) (idx : GridIndex (grid a)) :
    (launchOf a idx).head < a.q.heads := (idx ⟨1, by simp [grid, Grid.rank]⟩).isLt

theorem batch_lt (a : Arguments) (idx : GridIndex (grid a)) :
    (launchOf a idx).batch < a.batch := (idx ⟨0, by simp [grid, Grid.rank]⟩).isLt

/-- Logical tensor injectivity and nonoverlapping packed sequences imply
unique ownership across the complete three-axis batch/head grid, including tails. -/
theorem unique_writer (a : Arguments) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (ha : Admissible a r A s) (idx jdx : GridIndex (grid a))
    (i j : RealIO.Lane (layout a))
    (hi : RealIO.active (layout a) (launchOf a idx) (inputs s r) i)
    (hj : RealIO.active (layout a) (launchOf a jdx) (inputs s r) j)
    (he : RealIO.write (layout a) (launchOf a idx) (inputs s r) i =
      RealIO.write (layout a) (launchOf a jdx) (inputs s r) j) : idx = jdx ∧ i = j := by
  let p := layout a
  let x := launchOf a idx
  let y := launchOf a jdx
  let u := inputs s r
  have bx := batch_lt a idx
  have by' := batch_lt a jdx
  have hx := head_lt a idx
  have hy := head_lt a jdx
  have mx := ha.metadata _ bx
  have my := ha.metadata _ by'
  have dix : i.2.1.val < a.o.dims := by simpa [layout, ha.out_dims, ← ha.qk_dims] using hi.2
  have djx : j.2.1.val < a.o.dims := by simpa [layout, ha.out_dims, ← ha.qk_dims] using hj.2
  have sx : 0 ≤ u.start (x.batch) := mx.1
  have sy : 0 ≤ u.start (y.batch) := my.1
  have ex : u.start (x.batch) + (u.sequence (x.batch)-u.prompt (x.batch)) ≤ a.o.rows := mx.2.2.2.2.2.1
  have ey : u.start (y.batch) + (u.sequence (y.batch)-u.prompt (y.batch)) ≤ a.o.rows := my.2.2.2.2.2.1
  have ix : (queryRow p x i.1.val : Int) < u.sequence (x.batch)-u.prompt (x.batch) := hi.1
  have jy : (queryRow p y j.1.val : Int) < u.sequence (y.batch)-u.prompt (y.batch) := hj.1
  have hrow := ha.output_layout
    (u.start (x.batch)+queryRow p x i.1.val) (u.start (y.batch)+queryRow p y j.1.val)
    (x.head) (y.head) i.2.1.val j.2.1.val
    (by positivity) (by omega)
    (by positivity) (by omega)
    (by simpa [ha.out_heads] using hx) (by simpa [ha.out_heads] using hy) dix djx he
  have hp := ha.packed _ _ bx by' (queryRow p x i.1.val) (queryRow p y j.1.val)
    (by positivity) ix (by positivity) jy hrow.1
  have hrows : x.block*p.BM+i.1.val = y.block*p.BM+j.1.val := by
    exact_mod_cast hp.2
  have ib := i.1.isLt
  have jb := j.1.isLt
  have hblocks : x.block = y.block := by
    have bpos : 0 < p.BM := block_positive a
    have upperX : x.block*p.BM+i.1.val < (x.block+1)*p.BM := by nlinarith only [ib]
    have upperY : y.block*p.BM+j.1.val < (y.block+1)*p.BM := by nlinarith only [jb]
    apply Nat.le_antisymm
    · by_contra hn
      have hm := Nat.mul_le_mul_right p.BM (show y.block+1 ≤ x.block by omega)
      omega
    · by_contra hn
      have hm := Nat.mul_le_mul_right p.BM (show x.block+1 ≤ y.block by omega)
      omega
  refine ⟨launch_ext a hp.1 hrow.2.1 hblocks, ?_⟩
  apply Prod.ext
  · apply Fin.ext
    rw [hblocks] at hrows
    omega
  · apply Prod.ext
    · exact Fin.ext hrow.2.2
    · exact Subsingleton.elim _ _

end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Host
