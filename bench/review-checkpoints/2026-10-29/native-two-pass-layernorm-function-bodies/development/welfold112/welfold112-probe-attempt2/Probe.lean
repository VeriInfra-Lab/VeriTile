import VeriTile.Triton.DSL.CallSignature
namespace Welfold112Probe
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

def signature (B : Nat) (tree : DSL.WelfordSource.Tree B) : CallSignature where
  pointers := [("in_out_ptr0",.fp32),("in_out_ptr1",.fp32),("in_ptr0",.fp32),
    ("in_ptr1",.fp32),("in_ptr2",.fp32),("out_ptr0",.bf16)]
  scalars := [("xnumel",.i32),("rnumel",.i32)]
  constexpr := ["XBLOCK","RBLOCK"]
  readPorts := ["in_ptr0","in_ptr1","in_ptr2"]
  writePorts := ["in_out_ptr0","in_out_ptr1","out_ptr0"]
  orderedRowSums := [("_tmp3",⟨B,tree⟩),("_tmp12",⟨B,tree⟩)]

def kernel (in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 : RegionName)
    (xnumel rnumel RBLOCK : Nat) (tree : DSL.WelfordSource.Tree RBLOCK) : ComputeKernel :=
  let XBLOCK : Nat := 1
  with_triton_call (signature RBLOCK tree) call_args [xnumel := xnumel,rnumel := rnumel] in
  triton {
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
end
end Welfold112Probe
