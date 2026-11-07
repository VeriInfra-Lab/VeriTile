import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.DSL.WelfordSource
import VeriTile.Triton.DSL.OrderedSumCalls

/-! Literal repaired three-loop LayerNorm source. Runtime dimensions, pointer
storage, launch geometry and ordered-reduction metadata belong to the caller;
all source assignments, masks, casts, barriers and BF16 stores stay in order. -/
namespace VeriTile.Bench.Repairs.LayerNormWelfold.SourceCode
open VeriTile.Triton
noncomputable section

abbrev ReductionTree := DSL.WelfordSource.Tree
def blockCount (D B : Nat) : Nat := (D+B-1)/B

def callSignature : CallSignature where
  pointers := [("in_out_ptr0",.fp32),("in_out_ptr1",.fp32),("in_ptr0",.fp32),
    ("in_ptr1",.fp32),("in_ptr2",.fp32),("out_ptr0",.bf16)]
  scalars := [("xnumel",.integer),("rnumel",.integer)]
  constexpr := ["XBLOCK","RBLOCK"]
  readPorts := ["in_ptr0","in_ptr1","in_ptr2"]
  writePorts := ["in_out_ptr0","in_out_ptr1","out_ptr0"]

def sourceKernel (in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 : RegionName)
    (xnumel rnumel RBLOCK : Nat) (reductionTree : DSL.WelfordSource.Tree RBLOCK) : ComputeKernel :=
  let XBLOCK : Nat := 1
  with_triton_call callSignature call_args [xnumel := xnumel,rnumel := rnumel]
      ordered_sum_trees [tmp3 := reductionTree, tmp12 := reductionTree] in triton {
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tmp4
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = rnumel  # 4096.0
    tmp6 = tmp3 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, None)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 - tmp6
        tmp10 = tl.where(rmask, tmp9 * tmp9, 0.0)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tmp13
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp14 = rnumel  # 4096.0
    tmp15 = tmp12 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp18, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_first"
        ).to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp6
        tmp22 = tmp21 * tmp18
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp22 * tmp24
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tl.store(out_ptr0 + (r1 + (rnumel * x0)), tmp29, rmask)
  }

def candidateKernel (in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 : RegionName)
    (xnumel rnumel RBLOCK : Nat) (reductionTree : DSL.WelfordSource.Tree RBLOCK) : ComputeKernel :=
  let XBLOCK : Nat := 1
  with_triton_call callSignature call_args [xnumel := xnumel,rnumel := rnumel]
      ordered_sum_trees [tmp3 := reductionTree, tmp12 := reductionTree] in triton {
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tmp4
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = rnumel  # 4096.0
    tmp6 = tmp3 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, None)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 - tmp6
        tmp10 = tl.where(rmask, tmp9 * tmp9, 0.0)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tmp13
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp14 = rnumel  # 4096.0
    tmp15 = tmp12 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp18, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        row_offset = rnumel * x0
        tmp19 = tl.load(
            in_ptr0 + (r1 + row_offset), rmask, eviction_policy="evict_first"
        ).to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp6
        tmp22 = tmp21 * tmp18
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp22 * tmp24
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tl.store(out_ptr0 + (r1 + row_offset), tmp29, rmask)
  }

def rowKernel (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : DSL.WelfordSource.Tree B) : ComputeKernel :=
  sourceKernel Mean Rstd X W Bias Y S D B tree

def rowOffsetReusedKernel (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : DSL.WelfordSource.Tree B) : ComputeKernel :=
  candidateKernel Mean Rstd X W Bias Y S D B tree

/-- Select one of the two complete ordinary source bodies. -/
def selected (reused : Bool) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) : ComputeKernel :=
  if reused then rowOffsetReusedKernel X W Bias Y Mean Rstd S D B tree
  else rowKernel X W Bias Y Mean Rstd S D B tree

def runRows {α : Type} [Inhabited α] (M : FP.Structural.Algebra α) (kernel : ComputeKernel) :
    List Nat → FP.Structural.State α → Option (FP.Structural.State α)
  | [], s => some s
  | row :: rows, s => do
      let t ← FP.Structural.exec M kernel {s with pids := fun dim => if dim=0 then row else 0}
      runRows M kernel rows t

def launch {α : Type} [Inhabited α] (M : FP.Structural.Algebra α) (reused : Bool)
    (s : FP.Structural.State α) (Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) : Option (FP.Structural.State α) :=
  runRows M (selected reused "X" "W" "Bias" Y Mean Rstd S D B tree) (List.range S) s

end
end VeriTile.Bench.Repairs.LayerNormWelfold.SourceCode
