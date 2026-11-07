import VeriTile.Triton.DSL.WelfordCallMetadata
import bench.tritonbench_g.fused_layernorm_triton.Proofs.Support.SourceTypes
import VeriTile.Meta.StatementAudit

/-! Complete repaired Python and existing projection-hoisted candidate.
Caller integer/storage metadata and the selected helper tree stay outside
both source quotations. No source loop is peeled or flattened here. -/
namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Pasted
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 8000000

def callSignature : CallSignature where
  pointers := [("in_out_ptr0",.fp32),("in_ptr0",.fp32),("in_ptr1",.fp32),("in_ptr2",.fp32),("out_ptr0",.fp32),("out_ptr1",.bf16)]
  scalars := [("xnumel",.integer),("rnumel",.integer)]
  constexpr := ["XBLOCK","RBLOCK"]
  readPorts := ["in_ptr0","in_ptr1","in_ptr2"]
  writePorts := ["out_ptr0","in_out_ptr0","out_ptr1"]

def originalKernel (in_out_ptr0 in_ptr0 in_ptr1 in_ptr2 out_ptr0 out_ptr1 : RegionName)
    (xnumel rnumel RBLOCK : Nat) (tree : DSL.WelfordSource.Tree RBLOCK) : ComputeKernel :=
  let XBLOCK : Nat := 1
  with_triton_call callSignature call_args [xnumel := xnumel,rnumel := rnumel]
    welford_tree tree in triton {
    xnumel = tl.cast(xnumel, tl.int64)
    rnumel = tl.cast(rnumel, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), RBLOCK, tl.int64)
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK).to(tl.int64)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK).to(tl.int64)[None, :]
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(zero, rnumel, step):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_last"
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
    tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
        tmp3_mean, tmp3_m2, tmp3_weight, 1
    )
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp3, None)
    tmp6 = rnumel
    tmp7 = tmp4 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
    for roffset in range(zero, rnumel, step):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_first"
        ).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp18 = tl.load(in_ptr2 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 - tmp3
        tmp14 = tmp13 * tmp10
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (rnumel * x0)), tmp21, rmask)
  }

def hoistedKernel (in_out_ptr0 in_ptr0 in_ptr1 in_ptr2 out_ptr0 out_ptr1 : RegionName)
    (xnumel rnumel RBLOCK : Nat) (tree : DSL.WelfordSource.Tree RBLOCK) : ComputeKernel :=
  let XBLOCK : Nat := 1
  with_triton_call callSignature call_args [xnumel := xnumel,rnumel := rnumel]
    welford_tree tree in triton {
    xnumel = tl.cast(xnumel, tl.int64)
    rnumel = tl.cast(rnumel, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), RBLOCK, tl.int64)
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK).to(tl.int64)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK).to(tl.int64)[None, :]
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(zero, rnumel, step):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_last"
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
    tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
        tmp3_mean, tmp3_m2, tmp3_weight, 1
    )
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp3, None)
    tmp6 = rnumel
    tmp7 = tmp4 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
    # The single row axis is constant through normalization. Project once,
    # then use one-dimensional column tiles in every normalization chunk.
    xrow = tl.reshape(x0, ())
    row_mean = tl.reshape(tmp3, ())
    row_inverse = tl.reshape(tmp10, ())
    for roffset in range(zero, rnumel, step):
        rindex = roffset + tl.arange(0, RBLOCK).to(tl.int64)
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(
            in_ptr0 + (r1 + (rnumel * xrow)), rmask, eviction_policy="evict_first"
        ).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp18 = tl.load(in_ptr2 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 - row_mean
        tmp14 = tmp13 * row_inverse
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (rnumel * xrow)), tmp21, rmask)
  }

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Pasted

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Kernels
open VeriTile.Triton
noncomputable section

def kernel (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) : ComputeKernel :=
  Pasted.originalKernel Rstd X W Bias Mean Y S D B (sourceTree tree)

def hoisted (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) : ComputeKernel :=
  Pasted.hoistedKernel Rstd X W Bias Mean Y S D B (sourceTree tree)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Kernels
