import bench.repairs.tritonbench_g.context_attn_bloom.IO

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
open Host
noncomputable section
set_option maxHeartbeats 1000000

/-- The original tensor injectivity and packed-sequence argument, stated
on logical metadata so it applies equally to both actual source backends. -/
theorem unique_writer (a : Host.Arguments) (r : Regions) (extent : RegionBounds) (u : Math.Inputs)
    (ha : Conditions a r extent u) (idx jdx : GridIndex (grid a))
    (i j : RealIO.Lane (layout a))
    (hi : RealIO.active (layout a) (launchOf a idx) u i)
    (hj : RealIO.active (layout a) (launchOf a jdx) u j)
    (he : RealIO.write (layout a) (launchOf a idx) u i =
      RealIO.write (layout a) (launchOf a jdx) u j) : idx = jdx ∧ i = j := by
  let p := layout a
  let x := launchOf a idx
  let y := launchOf a jdx
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


#axiomsClean unique_writer
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO
