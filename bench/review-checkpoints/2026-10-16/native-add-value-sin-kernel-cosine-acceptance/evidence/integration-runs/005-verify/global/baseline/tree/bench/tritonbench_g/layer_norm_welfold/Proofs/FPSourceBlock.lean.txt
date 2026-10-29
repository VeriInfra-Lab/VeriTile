import bench.tritonbench_g.layer_norm_welfold.LayerNormWelfold
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.FPStages
namespace VeriTile.Bench.TritonBenchG.LayerNormWelfold.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

/-- Actual original full body at the autotuned XBLOCK=1 configuration.
All loop bodies, initial +0 accumulations, repeated casts and undefined
masked loads are retained. No 0-fill or variance mask is inserted. -/
def sourceTyped
    (in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 : RegionName)
    (xnumel rnumel XBLOCK RBLOCK : Nat) :
    ComputeKernel := triton {
  tl.region in_out_ptr0 = tl.float32, in_out_ptr1 = tl.float32, in_ptr0 = tl.float32, in_ptr1 = tl.float32, in_ptr2 = tl.float32, out_ptr0 = tl.float32
  xoffset = tl.program_id(0) * $(XBLOCK)
  xindex = xoffset + tl.arange(0, $(XBLOCK))[:, None]
  xmask = xindex < $(xnumel)
  rbase = tl.arange(0, $(RBLOCK))[None, :]
  x0 = xindex
  _tmp3 = tl.full([$(XBLOCK), $(RBLOCK)], 0, dtype=tl.float32)
  for roffset in range(0, $(rnumel), $(RBLOCK)) {
    rindex = roffset + rbase
    rmask = rindex < $(rnumel)
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp1 = (tmp0).to(tl.float32)
    tmp2, _ = tl.broadcast(tmp1, _tmp3)
    tmp4 = _tmp3 + tmp2
    _tmp3 = tmp4
  }
  tmp3_tmp = tl.sum(_tmp3, 1)
  tmp3 = tmp3_tmp[:, None]
  tmp5 = $(rnumel)
  tmp6 = tmp3 / tmp5
  tl.debug_barrier()
  tl.store(in_out_ptr0 + (x0), tmp6)
  _tmp12 = tl.full([$(XBLOCK), $(RBLOCK)], 0, dtype=tl.float32)
  for roffset in range(0, $(rnumel), $(RBLOCK)) {
    rindex = roffset + rbase
    rmask = rindex < $(rnumel)
    r1 = rindex
    tmp7 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp8 = (tmp7).to(tl.float32)
    tmp9 = tmp8 - tmp6
    tmp10 = tmp9 * tmp9
    tmp11, _ = tl.broadcast(tmp10, _tmp12)
    tmp13 = _tmp12 + tmp11
    _tmp12 = tmp13
  }
  tmp12_tmp = tl.sum(_tmp12, 1)
  tmp12 = tmp12_tmp[:, None]
  tmp14 = $(rnumel)
  tmp15 = tmp12 / tmp14
  tmp16 = 1e-05
  tmp17 = tmp15 + tmp16
  tmp18 = tl.rsqrt(tmp17)
  tl.debug_barrier()
  tl.store(in_out_ptr1 + (x0), tmp18)
  for roffset in range(0, $(rnumel), $(RBLOCK)) {
    rindex = roffset + rbase
    rmask = rindex < $(rnumel)
    r1 = rindex
    tmp19 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), rmask, eviction_policy="evict_first").to(tl.float32)
    tmp23 = tl.load(in_ptr1 + (r1), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp20 = (tmp19).to(tl.float32)
    tmp21 = tmp20 - tmp6
    tmp22 = tmp21 * tmp18
    tmp24 = (tmp23).to(tl.float32)
    tmp25 = tmp22 * tmp24
    tmp27 = (tmp26).to(tl.float32)
    tmp28 = tmp25 + tmp27
    tmp29 = (tmp28).to(tl.float32)
    tl.store(out_ptr0 + (r1 + ($(rnumel) * x0)), tmp29, rmask)
  }
}


/-- Specialize after parsing the symbolic source shape, so specialization
preserves the original broadcast syntax exactly. -/
def sourceOneBlock (mean rstd X W Bias Y : RegionName) (xnumel rnumel RBLOCK : Nat) : ComputeKernel :=
  sourceTyped mean rstd X W Bias Y xnumel rnumel 1 RBLOCK

def sourceColumn {RBLOCK : Nat} (i : TileIndex [1,RBLOCK]) : Nat := i.2.1.val

def sourceLoad {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (R : RegionName) (offset : Nat) (active : Prop) [Decidable active]
    {RBLOCK : Nat} (i : TileIndex [1,RBLOCK]) : α :=
  M.cast (some .fp32) .real .real
    (if active then (s.mem R offset).read .real else s.undef .real R offset)

def sourceMean {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X : RegionName) (rnumel RBLOCK : Nat) : α :=
  let xs : TileIndex [1,RBLOCK] → α := fun i =>
    M.binary (some .fp32) .real .add (M.literal (some .fp32) .real 0)
      (M.cast (some .fp32) .real .real
        (sourceLoad M s X (sourceColumn i + rnumel * s.pids 0) (sourceColumn i < rnumel) i))
  let total := M.reduceSum (some .fp32) (shape := [1,RBLOCK]) ⟨1, by simp⟩ Bool.false xs (⟨0, by decide⟩, PUnit.unit)
  M.binary (some .fp32) .real .div total (M.fromNat (some .fp32) rnumel)

def sourceRstd {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X : RegionName) (rnumel RBLOCK : Nat) (mean : α) : α :=
  let squared : TileIndex [1,RBLOCK] → α := fun i =>
    let x := M.cast (some .fp32) .real .real
      (sourceLoad M s X (sourceColumn i + rnumel * s.pids 0) (sourceColumn i < rnumel) i)
    let delta := M.binary (some .fp32) .real .sub x mean
    M.binary (some .fp32) .real .add (M.literal (some .fp32) .real 0)
      (M.binary (some .fp32) .real .mul delta delta)
  let total := M.reduceSum (some .fp32) (shape := [1,RBLOCK]) ⟨1, by simp⟩ Bool.false squared (⟨0, by decide⟩, PUnit.unit)
  let variance := M.binary (some .fp32) .real .div total (M.fromNat (some .fp32) rnumel)
  M.unary (some .fp32) .rsqrt
    (M.binary (some .fp32) .real .add variance (M.literal none .real (1e-05)))

def sourceY {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X W Bias : RegionName) (rnumel RBLOCK : Nat) (mean rstd : α) : TileIndex [1,RBLOCK] → α := fun i =>
  let active := sourceColumn i < rnumel
  let x := M.cast (some .fp32) .real .real
    (sourceLoad M s X (sourceColumn i + rnumel * s.pids 0) active i)
  let weight := M.cast (some .fp32) .real .real (sourceLoad M s W (sourceColumn i) active i)
  let bias := M.cast (some .fp32) .real .real (sourceLoad M s Bias (sourceColumn i) active i)
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .sub x mean) rstd) weight) bias)

def sourceMemory {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (mean rstd X W Bias Y : RegionName) (rnumel RBLOCK : Nat) : State α :=
  let avg := sourceMean M s X rnumel RBLOCK
  let afterMean := s.write mean (s.pids 0) (.mk .real avg)
  let inverse := sourceRstd M afterMean X rnumel RBLOCK avg
  let afterRstd := afterMean.write rstd (s.pids 0) (.mk .real inverse)
  (TileShape.allIndices [1,RBLOCK]).foldl (fun t i =>
    if sourceColumn i < rnumel then t.write Y (sourceColumn i + rnumel * s.pids 0)
      (.mk .real (sourceY M afterRstd X W Bias rnumel RBLOCK avg inverse i)) else t) afterRstd

theorem range_once {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (stop stride : Nat) (body : List ComputeStmt) (s : State α)
    (hpos : 0 < stop) (hle : stop ≤ stride) :
    FP.Structural.range M idx 0 stop stride body s = run M body (s.setReg idx .nat [] (fun _ => 0)) := by
  have hs : stride ≠ 0 := by omega
  have hstop : ¬ stride < stop := by omega
  rw [FP.Structural.range]
  simp only [hs, hpos, if_false, if_true, Nat.zero_add]
  cases hr : run M body (s.setReg idx .nat [] (fun _ => 0)) with
  | none => simp [hr]
  | some t => simp [hr, FP.Structural.range, hs, hstop]

specification source_one_block_fp32 {α : Type} [Inhabited α] (M : Algebra α)
    (mean rstd X W Bias Y : RegionName) (xnumel rnumel RBLOCK : Nat) (s : State α)
    (hpos : 0 < rnumel) (hle : rnumel ≤ RBLOCK) :
    ∃ final, FP.Structural.exec M (sourceOneBlock mean rstd X W Bias Y xnumel rnumel RBLOCK) s = some final ∧
      final.mem = (sourceMemory M s mean rstd X W Bias Y rnumel RBLOCK).mem := by
  cases mean; cases rstd; cases X; cases W; cases Bias; cases Y
  have hrange (idx : RegName) (body : List ComputeStmt) (t : State α) :
      FP.Structural.range M idx 0 rnumel RBLOCK body t =
        run M body (t.setReg idx .nat [] (fun _ => 0)) :=
    range_once M idx rnumel RBLOCK body t hpos hle
  simp [sourceOneBlock, sourceTyped, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, sourceMean, sourceRstd, sourceY,
    sourceLoad, sourceColumn, sourceMemory, hrange,
    TileShape.allIndices, List.finRange_succ, Fin.eq_zero, State.write, ofFloat, toFloat]
  all_goals first | rfl | (apply scatter_memory_congr; rfl)

#axiomsClean source_one_block_fp32
end
end VeriTile.Bench.TritonBenchG.LayerNormWelfold.FPStages
