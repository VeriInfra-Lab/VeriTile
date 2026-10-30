import bench.tritonbench_g.context_attn_fwd.Proofs.NativeAddresses

namespace VeriTile.Bench.Repairs.ContextAttnFwd.FormedBounds
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
open Geometry Native
noncomputable section
set_option maxHeartbeats 3000000

def queryPadded (c : Host.Call) (v : Geometry.Original.View3) : Geometry.Original.View3 :=
  {v with rows := v.rows+c.maxInput+128}
def cachePadded (v : Geometry.Original.View4) : Geometry.Original.View4 := {v with rows := v.rows+128}

/-- Exactly the wrapper's conservative formed-pointer sum (storage offset is
nonnegative), plus the separately checked scalar strides. -/
def Budget3 (v : Geometry.Original.View3) (storageOffset : Nat) : Prop :=
  storageOffset+(v.rows-1)*v.rowStride+(v.heads-1)*v.headStride+(v.dims-1)*v.dimStride<2^61 ∧
    v.rowStride<2^61 ∧ v.headStride<2^61 ∧ v.dimStride<2^61
def Budget4 (v : Geometry.Original.View4) (storageOffset : Nat) : Prop :=
  storageOffset+(v.batches-1)*v.batchStride+(v.rows-1)*v.rowStride+
    (v.heads-1)*v.headStride+(v.dims-1)*v.dimStride<2^61 ∧
    v.batchStride<2^61 ∧ v.rowStride<2^61 ∧ v.headStride<2^61 ∧ v.dimStride<2^61

structure Checked (c : Host.Call) : Prop where
  query : ∃ off,Budget3 (queryPadded c c.args.q) off
  output : ∃ off,Budget3 (queryPadded c c.args.o) off
  key : ∃ off,Budget4 (cachePadded c.args.k) off
  value : ∃ off,Budget4 (cachePadded c.args.v) off

theorem bound3 (v : Geometry.Original.View3) (off r h d : Nat) (hb : Budget3 v off)
    (hr : r<v.rows) (hh : h<v.heads) (hd : d<v.dims) :
    r*v.rowStride+h*v.headStride+d*v.dimStride<2^61 := by
  have ha := address3_lt v r h d hr hh hd
  have hb := hb.1
  unfold span3 at ha
  omega

theorem bound4 (v : Geometry.Original.View4) (off b r h d : Nat) (hb : Budget4 v off)
    (hbatch : b<v.batches) (hr : r<v.rows) (hh : h<v.heads) (hd : d<v.dims) :
    b*v.batchStride+r*v.rowStride+h*v.headStride+d*v.dimStride<2^61 := by
  have ha := address4_lt v b r h d hbatch hr hh hd
  have hb := hb.1
  unfold span4 at ha
  omega

theorem padded_query_row (c : Host.Call) (hc : Geometry.Checked c extent)
    (idx : GridIndex c.grid) (i : Fin c.layout.BM) :
    c.start (batchAt c idx)+Formula.queryRow c.layout (c.launchAt idx) i.val<
      c.args.q.rows+c.maxInput+128 := by
  have hmeta := hc.metadata _ (Host.batch_bound c idx)
  have hblock := block_start_bound c idx
  have hi := i.isLt
  have ht := tile c
  unfold Formula.queryRow
  simp only [batchAt] at *
  have hBM : c.layout.BM≤128 := by omega
  omega

theorem query_formed (c : Host.Call) (hc : Geometry.Checked c extent) (hf : Checked c)
    (idx : GridIndex c.grid) (i : Fin c.layout.BM) (d : Fin c.layout.D) :
    queryOffset c .query idx i d<2^61 ∧ queryOffset c .output idx i d<2^61 := by
  have hr := padded_query_row c hc idx i
  have hh := Host.head_bound c idx
  have hd : d.val<c.args.k.dims := d.isLt
  constructor
  · obtain ⟨off,hb⟩ := hf.query
    apply bound3 (queryPadded c c.args.q) off _ _ _ hb hr hh
    simpa only [queryPadded,hc.qk_dims] using hd
  · obtain ⟨off,hb⟩ := hf.output
    apply bound3 (queryPadded c c.args.o) off _ _ _ hb
    · simpa only [queryPadded,hc.output_rows] using hr
    · simpa only [queryPadded,hc.output_heads] using hh
    · simpa only [queryPadded,hc.output_dims] using hd

theorem cache_formed (c : Host.Call) (hc : Geometry.Checked c extent) (hf : Checked c)
    (idx : GridIndex c.grid) (step : Nat) (j : Fin c.layout.BN) (d : Fin c.layout.D)
    (ht : step<countAt c idx) :
    cacheOffset c .key idx step j d<2^61 ∧ cacheOffset c .value idx step j d<2^61 := by
  have hb := Host.batch_bound c idx
  have hmeta := hc.metadata _ hb
  have hp := padded_key_bound (c.launchAt idx).block c.layout.BM (seqAt c idx)
    (c.prompt (batchAt c idx)) step j (tile c) ht (by simpa only [c.blockN] using j.isLt)
  have hrk : step*c.layout.BN+j.val<c.args.k.rows+128 := by
    simp only [c.blockN]
    have htile := tile c
    have hBM : c.layout.BM≤128 := by omega
    simp only [seqAt,batchAt] at hp
    omega
  have hrv : step*c.layout.BN+j.val<c.args.v.rows+128 := by
    simp only [c.blockN]
    have htile := tile c
    have hBM : c.layout.BM≤128 := by omega
    simp only [seqAt,batchAt] at hp
    omega
  have hh := kv_head_bound c hc _ (Host.head_bound c idx)
  have hd : d.val<c.args.k.dims := d.isLt
  constructor
  · obtain ⟨off,hbudget⟩ := hf.key
    exact bound4 (cachePadded c.args.k) off _ _ _ _ hbudget (lt_of_lt_of_le hb hc.key_batches) hrk hh hd
  · obtain ⟨off,hbudget⟩ := hf.value
    apply bound4 (cachePadded c.args.v) off _ _ _ _ hbudget (lt_of_lt_of_le hb hc.value_batches) hrv
    · simpa only [cachePadded,hc.value_heads] using hh
    · simpa only [cachePadded,←hc.kv_dims] using hd

/-- All offset products, partial sums and byte scaling are signed64-safe.
The inequalities concern formed lanes, including masked-off lanes. -/
theorem sum3_transport (a b c : Nat) (h : a+b+c<2^61) :
    (word a).toInt=(a : Int) ∧ (word b).toInt=(b : Int) ∧ (word c).toInt=(c : Int) ∧
    (word (a+b)).toInt=((a+b : Nat) : Int) ∧
    (word (a+b+c)).toInt=((a+b+c : Nat) : Int) ∧
    (word (4*(a+b+c))).toInt=((4*(a+b+c) : Nat) : Int) := by
  exact ⟨signed _ (by omega),signed _ (by omega),signed _ (by omega),
    signed _ (by omega),signed _ (by omega),signed _ (by omega)⟩

theorem sum4_transport (a b c d : Nat) (h : a+b+c+d<2^61) :
    (word a).toInt=(a : Int) ∧ (word b).toInt=(b : Int) ∧ (word c).toInt=(c : Int) ∧
    (word d).toInt=(d : Int) ∧ (word (a+b)).toInt=((a+b : Nat) : Int) ∧
    (word (a+b+c)).toInt=((a+b+c : Nat) : Int) ∧
    (word (a+b+c+d)).toInt=((a+b+c+d : Nat) : Int) ∧
    (word (4*(a+b+c+d))).toInt=((4*(a+b+c+d) : Nat) : Int) := by
  exact ⟨signed _ (by omega),signed _ (by omega),signed _ (by omega),signed _ (by omega),
    signed _ (by omega),signed _ (by omega),signed _ (by omega),signed _ (by omega)⟩

end
end VeriTile.Bench.Repairs.ContextAttnFwd.FormedBounds
