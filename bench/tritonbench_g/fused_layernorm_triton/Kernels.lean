import VeriTile.Triton

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Kernels
open VeriTile.Triton
noncomputable section

/-- Source-selected binary helper tree; its order is numerical configuration. -/
inductive ReductionTree (N : Nat) where
  | leaf : Fin N → ReductionTree N
  | node : ReductionTree N → ReductionTree N → ReductionTree N

def ReductionTree.leaves {N : Nat} : ReductionTree N → List (Fin N)
  | .leaf i => [i]
  | .node l r => l.leaves ++ r.leaves

def ReductionTree.Complete {N : Nat} (t : ReductionTree N) : Prop :=
  t.leaves.Perm (List.finRange N)

/-- Syntax-only translation to the shared transparent helper expansion. -/
def sourceTree {N : Nat} : ReductionTree N → DSL.WelfordSource.Tree N
  | .leaf i => .leaf i
  | .node l r => .node (sourceTree l) (sourceTree r)

/-- Complete selected row source, with its first Welford iteration specialized.
The existing Int64 index projection applies to these natural addresses. -/
def kernel (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) : ComputeKernel := triton {
  tl.region X = tl.float32, W = tl.float32, Bias = tl.float32,
    Y = tl.bfloat16, Mean = tl.float32, Rstd = tl.float32
  xoffset = tl.program_id(0) * $(1)
  xindex = xoffset + tl.arange(0, $(1))[:, None]
  xmask = xindex < $(S)
  rbase = tl.arange(0, $(B))[None, :]
  x0 = xindex
  tmp3_mean = tl.zeros([$(1), $(B)], tl.float32)
  tmp3_m2 = tl.zeros([$(1), $(B)], tl.float32)
  tmp3_weight = tl.zeros([$(1), $(B)], tl.float32)
  roffset = $((0 : Nat))
  rindex = roffset + rbase
  rmask = rindex < $(D)
  r1 = rindex
  tmp0 = tl.load(X + (r1 + ($(D) * x0)), mask=rmask).to(tl.float32)
  tmp1 = (tmp0).to(tl.float32)
  tmp2, _ = tl.broadcast(tmp1, tmp3_mean)
  tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
    tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == $((0 : Nat)))
  tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
  tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
  tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
  for roffset in range($(B), $(D), $(B)) {
    rbase = tl.arange(0, $(B))[None, :]
    rindex = roffset + rbase
    rmask = rindex < $(D)
    r1 = rindex
    tmp0 = tl.load(X + (r1 + ($(D) * x0)), mask=rmask).to(tl.float32)
    tmp1 = (tmp0).to(tl.float32)
    tmp2, _ = tl.broadcast(tmp1, tmp3_mean)
    tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
      tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == $((0 : Nat)))
    tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
    tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
    tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
  }
  tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
    tmp3_mean, tmp3_m2, tmp3_weight, 1, tree=$(sourceTree tree))
  tmp3 = tmp3_tmp[:, None]
  tmp4 = tmp4_tmp[:, None]
  tmp5 = tmp5_tmp[:, None]
  tl.store(Mean + x0, tmp3)
  tmp6 = $(D)
  tmp7 = tmp4 / tl.toReal(tmp6)
  tmp8 = 1e-05
  tmp9 = tmp7 + tmp8
  tmp10 = tl.rsqrt(tmp9)
  tl.debug_barrier()
  tl.store(Rstd + x0, tmp10)
  for roffset in range($((0 : Nat)), $(D), $(B)) {
    xrow = tl.reshape(x0, [])
    rowMean = tl.reshape(tmp3, [])
    rowInverse = tl.reshape(tmp10, [])
    rindex = roffset + tl.arange(0, $(B))
    rmask = rindex < $(D)
    tmp11 = tl.load(X + (rindex + ($(D) * xrow)), mask=rmask).to(tl.float32)
    tmp15 = tl.load(W + rindex, mask=rmask).to(tl.float32)
    tmp18 = tl.load(Bias + rindex, mask=rmask).to(tl.float32)
    tmp12 = (tmp11).to(tl.float32)
    tmp13 = tmp12 - rowMean
    tmp14 = tmp13 * rowInverse
    tmp16 = (tmp15).to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = (tmp18).to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = (tmp20).to(tl.float32)
    stored = (tmp21).to(tl.bfloat16)
    tl.store(Y + (rindex + ($(D) * xrow)), stored, mask=rmask)
  }
}

/-- Hoist the row scalar projections out of the normalization loop.
This removes three repeated projections per chunk; no measured speedup is claimed. -/
def hoisted (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) : ComputeKernel := triton {
  tl.region X = tl.float32, W = tl.float32, Bias = tl.float32,
    Y = tl.bfloat16, Mean = tl.float32, Rstd = tl.float32
  xoffset = tl.program_id(0) * $(1)
  xindex = xoffset + tl.arange(0, $(1))[:, None]
  xmask = xindex < $(S)
  rbase = tl.arange(0, $(B))[None, :]
  x0 = xindex
  tmp3_mean = tl.zeros([$(1), $(B)], tl.float32)
  tmp3_m2 = tl.zeros([$(1), $(B)], tl.float32)
  tmp3_weight = tl.zeros([$(1), $(B)], tl.float32)
  roffset = $((0 : Nat))
  rindex = roffset + rbase
  rmask = rindex < $(D)
  r1 = rindex
  tmp0 = tl.load(X + (r1 + ($(D) * x0)), mask=rmask).to(tl.float32)
  tmp1 = (tmp0).to(tl.float32)
  tmp2, _ = tl.broadcast(tmp1, tmp3_mean)
  tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
    tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == $((0 : Nat)))
  tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
  tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
  tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
  for roffset in range($(B), $(D), $(B)) {
    rbase = tl.arange(0, $(B))[None, :]
    rindex = roffset + rbase
    rmask = rindex < $(D)
    r1 = rindex
    tmp0 = tl.load(X + (r1 + ($(D) * x0)), mask=rmask).to(tl.float32)
    tmp1 = (tmp0).to(tl.float32)
    tmp2, _ = tl.broadcast(tmp1, tmp3_mean)
    tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
      tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == $((0 : Nat)))
    tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
    tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
    tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
  }
  tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
    tmp3_mean, tmp3_m2, tmp3_weight, 1, tree=$(sourceTree tree))
  tmp3 = tmp3_tmp[:, None]
  tmp4 = tmp4_tmp[:, None]
  tmp5 = tmp5_tmp[:, None]
  tl.store(Mean + x0, tmp3)
  tmp6 = $(D)
  tmp7 = tmp4 / tl.toReal(tmp6)
  tmp8 = 1e-05
  tmp9 = tmp7 + tmp8
  tmp10 = tl.rsqrt(tmp9)
  tl.debug_barrier()
  tl.store(Rstd + x0, tmp10)
  xrow = tl.reshape(x0, [])
  rowMean = tl.reshape(tmp3, [])
  rowInverse = tl.reshape(tmp10, [])
  for roffset in range($((0 : Nat)), $(D), $(B)) {
    rindex = roffset + tl.arange(0, $(B))
    rmask = rindex < $(D)
    tmp11 = tl.load(X + (rindex + ($(D) * xrow)), mask=rmask).to(tl.float32)
    tmp15 = tl.load(W + rindex, mask=rmask).to(tl.float32)
    tmp18 = tl.load(Bias + rindex, mask=rmask).to(tl.float32)
    tmp12 = (tmp11).to(tl.float32)
    tmp13 = tmp12 - rowMean
    tmp14 = tmp13 * rowInverse
    tmp16 = (tmp15).to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = (tmp18).to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = (tmp20).to(tl.float32)
    stored = (tmp21).to(tl.bfloat16)
    tl.store(Y + (rindex + ($(D) * xrow)), stored, mask=rmask)
  }
}

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Kernels
