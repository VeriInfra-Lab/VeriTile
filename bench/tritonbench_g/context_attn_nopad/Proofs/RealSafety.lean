import bench.tritonbench_g.context_attn_nopad.Proofs.RealData
import bench.tritonbench_g.context_attn_nopad.Proofs.IOBounds
import bench.tritonbench_g.context_attn_nopad.Proofs.Mathematical_WholeSafety

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 2500000

private theorem count_bound (c : Call) (l : Formula.Launch) (a : Formula.Inputs) :
    Formula.count c.layout l a≤l.block+1 := by
  unfold Formula.count
  split
  · simp [Call.layout]
  · omega

/-- Every actual signed source address is bounded by its logical IO channel.
Only live windows are required, including the aligned full causal key loop. -/
theorem source_bounds (c : Call) (r : Regions) (xs : Values c) (s : BlockState)
    (hmeta : Metadata c r s) (bounds : RegionBounds) (hb : (io c r).InBounds bounds s xs)
    (idx : GridIndex c.grid) :
    AddressBounds.Valid c.layout r (c.launch idx) (Program.inputs s r) bounds := by
  have hs := hmeta.start _ (launch_batch_lt c idx)
  have hl := hmeta.length _ (launch_batch_lt c idx)
  constructor
  · rw [hs]; exact Int.natCast_nonneg _
  · exact hb.1 3 ⟨_,launch_batch_lt c idx⟩ trivial
  · exact hb.1 4 ⟨_,launch_batch_lt c idx⟩ trivial
  · intro i d on
    have ht : (c.launch idx).block*128+i.val<c.length (c.launch idx).batch := by
      simpa only [Formula.sequence,hl,Formula.queryRow,Call.layout,Int.ofNat_lt] using on
    let j := IOData.coordinate c idx _ (IOData.row_bound c idx i) d
    have hin : (io c r).imask (⟨0,by decide⟩ : Fin 5) xs (s.pids 0) (s.pids 1) (s.pids 2) (encode c j) := by
      change IOData.active c (decode c (encode c j))
      simpa only [decode_encode] using ht
    have h := hb.1 (⟨0,by decide⟩ : Fin 5) (encode c j) hin
    simp [io,read,inputRegion] at h
    change c.q.offset (c.start (c.launch idx).batch+((c.launch idx).block*128+i.val)) (c.launch idx).head d.val<bounds r.query at h
    simp only [AddressBounds.inside,AddressBounds.queryAddress,hs,Formula.queryRow,Call.layout]
    constructor
    · positivity
    · exact_mod_cast h
  · intro n hn k d on
    have ht : n*128+k.val<c.length (c.launch idx).batch := by
      simpa only [Formula.sequence,hl,Call.layout,Int.ofNat_lt] using on
    have hn' : n*128<((c.launch idx).block+1)*128 := by
      have := count_bound c (c.launch idx) (Program.inputs s r)
      omega
    let j := IOData.coordinate c idx _ (IOData.aligned_bound c idx n hn' k) d
    have hin : (io c r).imask (⟨1,by decide⟩ : Fin 5) xs (s.pids 0) (s.pids 1) (s.pids 2) (encode c j) := by
      change IOData.active c (decode c (encode c j))
      simpa only [decode_encode] using ht
    have h := hb.1 (⟨1,by decide⟩ : Fin 5) (encode c j) hin
    simp [io,read,inputRegion] at h
    change c.k.offset (c.start (c.launch idx).batch+(n*128+k.val)) (c.launch idx).head d.val<bounds r.key at h
    simp only [AddressBounds.inside,AddressBounds.keyAddress,hs,Call.layout]
    constructor
    · positivity
    · exact_mod_cast h
  · intro n hn k d on
    have ht : n*128+k.val<c.length (c.launch idx).batch := by
      simpa only [Formula.sequence,hl,Call.layout,Int.ofNat_lt] using on
    have hn' : n*128<((c.launch idx).block+1)*128 := by
      have := count_bound c (c.launch idx) (Program.inputs s r)
      omega
    let j := IOData.coordinate c idx _ (IOData.aligned_bound c idx n hn' k) d
    have hin : (io c r).imask (⟨2,by decide⟩ : Fin 5) xs (s.pids 0) (s.pids 1) (s.pids 2) (encode c j) := by
      change IOData.active c (decode c (encode c j))
      simpa only [decode_encode] using ht
    have h := hb.1 (⟨2,by decide⟩ : Fin 5) (encode c j) hin
    simp [io,read,inputRegion] at h
    change c.v.offset (c.start (c.launch idx).batch+(n*128+k.val)) (c.launch idx).head d.val<bounds r.value at h
    simp only [AddressBounds.inside,AddressBounds.valueAddress,hs,Call.layout]
    constructor
    · positivity
    · exact_mod_cast h
  · intro i d on
    have ht : (c.launch idx).block*128+i.val<c.length (c.launch idx).batch := by
      simpa only [Formula.sequence,hl,Formula.queryRow,Call.layout,Int.ofNat_lt] using on
    let j := IOData.coordinate c idx _ (IOData.row_bound c idx i) d
    have hin : (io c r).omask 0 xs (s.pids 0) (s.pids 1) (s.pids 2) (encode c j) := by
      simpa only [io,decode_encode] using ht
    have h := hb.2.1 0 (encode c j) hin
    simp only [io,decode_encode] at h
    change c.o.offset (c.start (c.launch idx).batch+((c.launch idx).block*128+i.val)) (c.launch idx).head d.val<bounds r.out at h
    simp only [AddressBounds.inside,AddressBounds.outputAddress,hs,Formula.queryRow,Call.layout]
    constructor
    · positivity
    · exact_mod_cast h

theorem grid_safe (c : Call) (r : Regions) (xs : Values c) (s : BlockState)
    (hmeta : Metadata c r s) (bounds : RegionBounds) (hb : (io c r).InBounds bounds s xs)
    (idx : GridIndex c.grid) :
    Kernel.TraceSafe bounds (kernel c r).eraseDType.toAlgKernel (s.withGridIndex idx) := by
  apply WholeSafety.complete c.layout r .fp32 Bool.false c.scale (s.withGridIndex idx) (by norm_num [Call.layout])
  rw [launch_grid,inputs_grid]
  exact source_bounds c r xs s hmeta bounds hb idx

#axiomsClean grid_safe
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData
