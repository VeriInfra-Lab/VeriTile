import bench.tritonbench_g.attn_fwd_triton.Proofs.SafetyLoop
import bench.tritonbench_g.attn_fwd_triton.Proofs.Real

/-! Ordinary valid-view and sufficient-scale allocation facts imply every
local access bound. No trace, successful execution or output law is assumed. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ViewBounds
open VeriTile.Triton Memory AccessSafety
noncomputable section

def View (bounds : RegionBounds) (r : RegionName) (B H N D sz sh sn sd offset : Nat) : Prop :=
  ∀ z < B, ∀ h < H, ∀ n < N, ∀ d < D, RealProofs.address sz sh sn sd z h n d+offset < bounds r

structure Windows (bounds : RegionBounds) (p : Parameters) (offsets : BaseOffsets := {}) : Prop where
  query : View bounds p.Q p.batches p.heads p.context p.headDim p.queryBatch p.queryHead p.queryRow p.queryDim offsets.query
  key : View bounds p.K p.batches p.heads p.context p.headDim p.keyBatch p.keyHead p.keyRow p.keyDim offsets.key
  value : View bounds p.V p.batches p.heads p.context p.headDim p.valueBatch p.valueHead p.valueRow p.valueDim offsets.value
  output : View bounds p.Out p.batches p.heads p.context p.headDim p.outputBatch p.outputHead p.outputRow p.outputDim offsets.output
  queryScale : p.batches*p.heads*((p.context+127)/128)+offsets.queryScale ≤ bounds p.QS
  keyScale : p.batches*p.heads*((p.context+63)/64)+offsets.keyScale ≤ bounds p.KS

structure Program (c : Config) : Prop where
  batch : c.z < c.p.batches
  head : c.h < c.p.heads
  row : c.startM*128 < c.p.context

theorem query (bounds : RegionBounds) (c : Config) (BD : Nat) (hv : Windows bounds c.p c.offsets) (hp : Program c) :
    QueryBound bounds c 128 BD := by
  intro r d hr hd
  simpa only [RealProofs.address,qbase,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
    hv.query c.z hp.batch c.h hp.head (c.startM*128+r.val) hr d.val hd

theorem key (bounds : RegionBounds) (c : Config) (BD start : Nat) (hv : Windows bounds c.p c.offsets) (hp : Program c) :
    KeyBound bounds c 64 BD start := by
  intro j d hj hd
  simpa only [RealProofs.address,KeyBound,kbase,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
    hv.key c.z hp.batch c.h hp.head (start+j.val) hj d.val hd

theorem value (bounds : RegionBounds) (c : Config) (BD start : Nat) (hv : Windows bounds c.p c.offsets) (hp : Program c) :
    ValueBound bounds c 64 BD start := by
  intro j d hj hd
  simpa only [RealProofs.address,vbase,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
    hv.value c.z hp.batch c.h hp.head (start+j.val) hj d.val hd

theorem output (bounds : RegionBounds) (c : Config) (BD : Nat) (hv : Windows bounds c.p c.offsets) (hp : Program c) :
    OutputBound bounds c 128 BD := by
  intro i hi
  have hh : c.startM*128+i.1.val < c.p.context ∧ i.2.1.val < c.p.headDim := by
    simpa [Output.active,Output.outputConfig,VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealEpilogue.active] using hi
  simpa only [Output.address,Output.outputConfig,VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealEpilogue.address,
    RealProofs.address,obase,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
    hv.output c.z hp.batch c.h hp.head _ hh.1 _ hh.2

theorem packed (B H z h count block : Nat) (hz : z < B) (hh : h < H) (hb : block < count) :
    (z*H+h)*count+block < B*H*count := by
  have hs := Nat.mul_le_mul_right count (Nat.succ_le_of_lt (RealProofs.head_grid_coverage B H z h hz hh))
  simp only [Nat.succ_mul] at hs
  omega

theorem query_scale (bounds : RegionBounds) (c : Config) (hv : Windows bounds c.p c.offsets) (hp : Program c) :
    qsbase c 128+c.startM < bounds c.p.QS := by
  have hb : c.startM < (c.p.context+127)/128 := by have := hp.row; omega
  have h := packed c.p.batches c.p.heads c.z c.h _ c.startM hp.batch hp.head hb
  unfold qsbase
  have he : (c.p.context+128-1)/128 = (c.p.context+127)/128 := by omega
  rw [he]
  have := Nat.add_lt_add_right h c.offsets.queryScale
  have := hv.queryScale
  omega

theorem key_scale (bounds : RegionBounds) (c : Config) (start : Nat) (hv : Windows bounds c.p c.offsets)
    (hp : Program c) (hs : start < c.p.context) : ksbase c 64+start/64 < bounds c.p.KS := by
  have h := packed c.p.batches c.p.heads c.z c.h _ (start/64) hp.batch hp.head
    (RealProofs.scale_tail_bound c.p.context start hs)
  unfold ksbase
  have he : (c.p.context+64-1)/64 = (c.p.context+63)/64 := by omega
  rw [he]
  have := Nat.add_lt_add_right h c.offsets.keyScale
  have := hv.keyScale
  omega

theorem loop_bounds (bounds : RegionBounds) (c : Config) (BD lower upper : Nat)
    (hv : Windows bounds c.p c.offsets) (hp : Program c) (hupper : upper ≤ c.p.context) :
    SafetyLoop.Bounds bounds c 64 BD lower upper := by
  refine ⟨fun _ _ => key bounds c BD _ hv hp,?_,fun _ _ => value bounds c BD _ hv hp⟩
  intro i hi
  apply key_scale bounds c _ hv hp
  exact Nat.lt_of_lt_of_le ((VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoop.cursor_lt lower upper 64 i (by omega)).mpr hi) hupper

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ViewBounds
