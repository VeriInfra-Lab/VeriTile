import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Host
import bench.tritonbench_g.context_attn_fwd.Proofs.HostInterface

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Geometry
open VeriTile.Triton
namespace Original
abbrev View3 := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.View3
abbrev View4 := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.View4
end Original
noncomputable section
set_option maxHeartbeats 3000000

/-- Checked element spans from ordinary tensor shape/stride metadata. -/
def span3 (v : Original.View3) : Nat :=
  (v.rows-1)*v.rowStride+(v.heads-1)*v.headStride+(v.dims-1)*v.dimStride+1

def span4 (v : Original.View4) : Nat :=
  (v.batches-1)*v.batchStride+(v.rows-1)*v.rowStride+
    (v.heads-1)*v.headStride+(v.dims-1)*v.dimStride+1

theorem address3_lt (v : Original.View3) (r h d : Nat)
    (hr : r<v.rows) (hh : h<v.heads) (hd : d<v.dims) :
    r*v.rowStride+h*v.headStride+d*v.dimStride<span3 v := by
  have a := Nat.mul_le_mul_right v.rowStride (show r≤v.rows-1 by omega)
  have b := Nat.mul_le_mul_right v.headStride (show h≤v.heads-1 by omega)
  have c := Nat.mul_le_mul_right v.dimStride (show d≤v.dims-1 by omega)
  unfold span3
  omega

theorem address4_lt (v : Original.View4) (b r h d : Nat)
    (hb : b<v.batches) (hr : r<v.rows) (hh : h<v.heads) (hd : d<v.dims) :
    b*v.batchStride+r*v.rowStride+h*v.headStride+d*v.dimStride<span4 v := by
  have a := Nat.mul_le_mul_right v.batchStride (show b≤v.batches-1 by omega)
  have c := Nat.mul_le_mul_right v.rowStride (show r≤v.rows-1 by omega)
  have e := Nat.mul_le_mul_right v.headStride (show h≤v.heads-1 by omega)
  have f := Nat.mul_le_mul_right v.dimStride (show d≤v.dims-1 by omega)
  unfold span4
  omega

/-- The ordinary contiguous Tensor metadata rule, including singleton strides. -/
def Contiguous (v : Original.View3) : Prop :=
  (1<v.rows → v.rowStride=v.heads*v.dims) ∧
  (0<v.rows → 1<v.heads → v.headStride=v.dims) ∧
  (0<v.rows → 0<v.heads → 1<v.dims → v.dimStride=1)

theorem contiguous_address (v : Original.View3) (hv : Contiguous v) (r h d : Nat)
    (hr : r<v.rows) (hh : h<v.heads) (hd : d<v.dims) :
    r*v.rowStride+h*v.headStride+d*v.dimStride=(r*v.heads+h)*v.dims+d := by
  have rows : r*v.rowStride=r*(v.heads*v.dims) := by
    by_cases he : 1<v.rows
    · rw [hv.1 he]
    · have : r=0 := by omega
      simp [this]
  have heads : h*v.headStride=h*v.dims := by
    by_cases he : 1<v.heads
    · rw [hv.2.1 (by omega) he]
    · have : h=0 := by omega
      simp [this]
  have dims : d*v.dimStride=d := by
    by_cases he : 1<v.dims
    · rw [hv.2.2 (by omega) (by omega) he,Nat.mul_one]
    · have : d=0 := by omega
      simp [this]
  rw [rows,heads,dims]
  ring

theorem contiguous_injective (v : Original.View3) (hv : Contiguous v) : v.Injective := by
  intro x y h k d e hx hxr hy hyr hh hk hd he eq
  have hn : 0<v.dims := by omega
  have hhpos : 0<v.heads := by omega
  have xr : x.toNat<v.rows := by omega
  have yr : y.toNat<v.rows := by omega
  have ax : (v.address x h d).toNat=x.toNat*v.rowStride+h*v.headStride+d*v.dimStride := by
    unfold VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.View3.address
    rw [←Int.toNat_of_nonneg hx]
    norm_cast
  have ay : (v.address y k e).toNat=y.toNat*v.rowStride+k*v.headStride+e*v.dimStride := by
    unfold VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.View3.address
    rw [←Int.toNat_of_nonneg hy]
    norm_cast
  rw [ax,ay,contiguous_address v hv _ _ _ xr hh hd,
    contiguous_address v hv _ _ _ yr hk he] at eq
  have dims : d=e := by
    have hm := congrArg (· % v.dims) eq
    simpa [Nat.add_mod,Nat.mod_eq_of_lt hd,Nat.mod_eq_of_lt he] using hm
  have blocks : x.toNat*v.heads+h=y.toNat*v.heads+k := by
    have hm := congrArg (· / v.dims) eq
    simpa [Nat.add_div,hn,Nat.ne_of_gt hn,Nat.div_eq_of_lt hd,Nat.div_eq_of_lt he,Nat.mod_eq_of_lt hd,Nat.mod_eq_of_lt he,Nat.not_le.mpr hd,Nat.not_le.mpr he] using hm
  have heads : h=k := by
    have hm := congrArg (· % v.heads) blocks
    simpa [Nat.add_mod,Nat.mod_eq_of_lt hh,Nat.mod_eq_of_lt hk] using hm
  have rows : x.toNat=y.toNat := by
    have hm := congrArg (· / v.heads) blocks
    simpa [Nat.add_div,hhpos,Nat.ne_of_gt hhpos,Nat.div_eq_of_lt hh,Nat.div_eq_of_lt hk,Nat.mod_eq_of_lt hh,Nat.mod_eq_of_lt hk,Nat.not_le.mpr hh,Nat.not_le.mpr hk] using hm
  exact ⟨by omega,heads,dims⟩

/-- These are the wrapper's checked shape, packed-row, capacity and span facts.
Actual source-address and writer predicates are consequences, not fields. -/
structure Checked (c : Host.Call) (extent : RegionBounds) : Prop where
  kv_positive : 0<c.args.k.heads
  grouped : c.args.k.heads ∣ c.args.q.heads
  qk_dims : c.args.q.dims=c.args.k.dims
  kv_dims : c.args.k.dims=c.args.v.dims
  output_rows : c.args.o.rows=c.args.q.rows
  output_heads : c.args.o.heads=c.args.q.heads
  output_dims : c.args.o.dims=c.args.k.dims
  value_heads : c.args.v.heads=c.args.k.heads
  key_batches : c.batches≤c.args.k.batches
  value_batches : c.batches≤c.args.v.batches
  width : c.args.k.dims ∈ ({16,32,64,128,256} : Finset Nat)
  metadata : ∀ b,b<c.batches →
    c.length b≤c.maxInput ∧ c.start b+c.length b≤c.args.q.rows ∧
    c.prompt b+c.length b≤c.args.k.rows ∧ c.prompt b+c.length b≤c.args.v.rows
  packed : ∀ b z,b<c.batches → z<c.batches → 0<c.length b → 0<c.length z → b≠z →
    c.start b+c.length b≤c.start z ∨ c.start z+c.length z≤c.start b
  contiguous : Contiguous c.args.o
  query_span : 0<c.args.q.rows → 0<c.args.q.heads → 0<c.args.q.dims → span3 c.args.q≤extent c.regions.query
  key_span : 0<c.args.k.batches → 0<c.args.k.rows → 0<c.args.k.heads → 0<c.args.k.dims → span4 c.args.k≤extent c.regions.key
  value_span : 0<c.args.v.batches → 0<c.args.v.rows → 0<c.args.v.heads → 0<c.args.v.dims → span4 c.args.v≤extent c.regions.value
  output_span : 0<c.args.o.rows → 0<c.args.o.heads → 0<c.args.o.dims → span3 c.args.o≤extent c.regions.out
  start_span : c.batches≤extent c.regions.start
  sequence_span : c.batches≤extent c.regions.sequence
  prompt_span : c.batches≤extent c.regions.prompt

theorem group_positive (c : Host.Call) (hc : Checked c extent) : 0<c.layout.kvGroup := by
  exact Nat.div_pos (Nat.le_of_dvd c.headsPositive hc.grouped) hc.kv_positive

theorem kv_head_bound (c : Host.Call) (hc : Checked c extent) (h : Nat) (hh : h<c.layout.heads) :
    h/c.layout.kvGroup<c.args.k.heads := by
  apply (Nat.div_lt_iff_lt_mul (group_positive c hc)).mpr
  have he : c.args.k.heads*(c.args.q.heads/c.args.k.heads)=c.args.q.heads :=
    Nat.mul_div_cancel' hc.grouped
  simpa [Host.Call.layout,VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.layout,he] using hh

end
end VeriTile.Bench.Repairs.ContextAttnFwd.Geometry
