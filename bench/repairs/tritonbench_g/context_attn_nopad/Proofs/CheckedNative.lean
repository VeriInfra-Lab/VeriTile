import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.CheckedBridge

namespace VeriTile.Bench.Repairs.ContextAttnNopad.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 3000000

def padded (c : Call) (requestedMax : Nat) : VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.View3 where
  rows := c.q.rows+requestedMax+128
  heads := c.q.heads
  dims := c.q.dims
  rowStride := c.q.heads*c.q.dims
  headStride := c.q.dims
  dimStride := 1

theorem budget (c : Call) (requestedMax : Nat) (v : View) (h : formedBudget c requestedMax v) :
    VeriTile.Bench.Repairs.ContextAttnFwd.FormedBounds.Budget3 (padded c requestedMax) v.storageOffset := by
  refine ⟨?_,h.2.1,h.2.2,by norm_num [padded]⟩
  simpa only [padded,Nat.mul_one] using h.1

theorem padded_row (c : Call) (r : Regions) (extent : RegionBounds) (requestedMax : Nat)
    (g : geometryChecks c r extent) (limit : c.maxInputLen≤requestedMax)
    (b block lane : Nat) (hb : b < c.batches) (hblock : block < c.blocks) (hlane : lane < 128) :
    c.start b+(block*128+lane) < c.q.rows+requestedMax+128 := by
  have capacity := (g.2.2.2.2.2.2.1 ⟨b,hb⟩).1
  change c.start b+c.length b≤c.q.rows at capacity
  change block < (c.maxInputLen+127)/128 at hblock
  omega

/-- Includes every query/output/key/value lane formed in the executed tile,
including false-mask lanes. Guard sums bound all element and byte products. -/
theorem formed_offset (c : Call) (r : Regions) (extent : RegionBounds) (requestedMax : Nat)
    (g : geometryChecks c r extent) (limit : c.maxInputLen≤requestedMax)
    (v : View) (h : formedBudget c requestedMax v)
    (b block lane head dim : Nat) (hb : b < c.batches) (ht : block < c.blocks)
    (hl : lane < 128) (hh : head < c.q.heads) (hd : dim < c.q.dims) :
    c.q.offset (c.start b+(block*128+lane)) head dim < 2^61 := by
  have bound := VeriTile.Bench.Repairs.ContextAttnFwd.FormedBounds.bound3 (padded c requestedMax) v.storageOffset
    (c.start b+(block*128+lane)) head dim (budget c requestedMax v h)
    (padded_row c r extent requestedMax g limit b block lane hb ht hl) hh hd
  have canonical := g.2.2.2.2.2.1
  simpa only [Tensor3.offset,canonical.1,canonical.2.1,canonical.2.2,padded] using bound

theorem all_formed_exact (c : Call) (r : Regions) (extent : RegionBounds) (requestedMax : Nat)
    (g : geometryChecks c r extent) (limit : c.maxInputLen≤requestedMax)
    (v : View) (h : formedBudget c requestedMax v)
    (b block iter lane head dim : Nat) (hb : b < c.batches) (ht : block < c.blocks)
    (hi : iter≤block) (hl : lane < 128) (hh : head < c.q.heads) (hd : dim < c.q.dims) :
    (NativeProgram.queryWord c.q (c.start b) block lane head dim).toInt=
      (c.q.offset (c.start b+(block*128+lane)) head dim : Int) ∧
    (NativeProgram.keyWord c.q (c.start b) (iter*128) lane head dim).toInt=
      (c.q.offset (c.start b+iter*128+lane) head dim : Int) ∧
    (NativeProgram.word (4*c.q.offset (c.start b+(block*128+lane)) head dim)).toInt=
      (4*c.q.offset (c.start b+(block*128+lane)) head dim : Nat) ∧
    (NativeProgram.word (4*c.q.offset (c.start b+iter*128+lane) head dim)).toInt=
      (4*c.q.offset (c.start b+iter*128+lane) head dim : Nat) := by
  have q := formed_offset c r extent requestedMax g limit v h b block lane head dim hb ht hl hh hd
  have k := formed_offset c r extent requestedMax g limit v h b iter lane head dim hb (lt_of_le_of_lt hi ht) hl hh hd
  rw [← Nat.add_assoc] at k
  refine ⟨?_,?_,NativeProgram.word_exact (by omega),NativeProgram.word_exact (by omega)⟩
  · rw [NativeProgram.queryWord_eq]; exact NativeProgram.word_exact (by omega)
  · rw [NativeProgram.keyWord_eq]; exact NativeProgram.word_exact (by omega)

#axiomsClean formed_offset
#axiomsClean all_formed_exact
end
end VeriTile.Bench.Repairs.ContextAttnNopad.Checked
