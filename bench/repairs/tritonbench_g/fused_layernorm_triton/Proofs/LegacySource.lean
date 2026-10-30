import bench.repairs.tritonbench_g.fused_layernorm_triton.Kernels

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.LegacySource
open VeriTile.Triton
noncomputable section

def welfordPrefix (X : RegionName) (xnumel rnumel RBLOCK : Nat) : List ComputeStmt := (triton {
  tl.region X = tl.float32
  xoffset = tl.program_id(0) * $(1)
  xindex = xoffset + tl.arange(0, $(1))[:, None]
  xmask = xindex < $(xnumel)
  rbase = tl.arange(0, $(RBLOCK))[None, :]
  x0 = xindex
  tmp3_mean = tl.zeros([$(1), $(RBLOCK)], tl.float32)
  tmp3_m2 = tl.zeros([$(1), $(RBLOCK)], tl.float32)
  tmp3_weight = tl.zeros([$(1), $(RBLOCK)], tl.float32)
  roffset = $((0 : Nat))
  rindex = roffset + rbase
  rmask = rindex < $(rnumel)
  r1 = rindex
  tmp0 = tl.load(X + (r1 + ($(rnumel) * x0)), mask=rmask).to(tl.float32)
  tmp1 = (tmp0).to(tl.float32)
  tmp2, _ = tl.broadcast(tmp1, tmp3_mean)
  tmp3_weight_next = tl.full([$(1), $(RBLOCK)], $((1 : ℝ)), dtype=tl.float32)
  tmp3_mean_next = tmp2
  tmp3_m2_next = tl.zeros([$(1), $(RBLOCK)], tl.float32)
  tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
  tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
  tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
}).surfaceBody

def nextChunk (X : RegionName) (D B k : Nat) : List ComputeStmt :=
  ((triton {
    tl.region X = tl.float32
    x0 = tl.program_id(0) + tl.arange(0, $(1))[:, None]
    tmp3_mean = tl.zeros([$(1), $(B)], tl.float32)
    tmp3_m2 = tl.zeros([$(1), $(B)], tl.float32)
    tmp3_weight = tl.zeros([$(1), $(B)], tl.float32)
    roffset = $(k * B)
    rbase = tl.arange(0, $(B))[None, :]
    rindex = roffset + rbase
    rmask = rindex < $(D)
    r1 = rindex
    tmp0 = tl.load(X + (r1 + ($(D) * x0)), mask=rmask).to(tl.float32)
    tmp1 = (tmp0).to(tl.float32)
    tmp2, _ = tl.broadcast(tmp1, tmp3_mean)
    delta = tmp2 - tmp3_mean
    tmp3_weight_next = tmp3_weight + 1.0
    tmp3_mean_next = tmp3_mean + delta / tmp3_weight_next
    delta2 = tmp2 - tmp3_mean_next
    tmp3_m2_next = tmp3_m2 + delta * delta2
    tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
    tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
    tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
  } : ComputeKernel).surfaceBody).drop 4

def chunkCode (X : RegionName) (D B : Nat) : Nat → Nat → List ComputeStmt
  | _, 0 => []
  | start, count + 1 => nextChunk X D B start ++ chunkCode X D B (start + 1) count

def welfordSuffix (Rstd Mean : RegionName) (rnumel : Nat) : List ComputeStmt :=
  ((triton {
    tl.region Rstd = tl.float32, Mean = tl.float32
    x0 = tl.program_id(0) + tl.arange(0, $(1))[:, None]
    tmp3_tmp = tl.zeros([$(1)], dtype=tl.float32)
    tmp4_tmp = tl.zeros([$(1)], dtype=tl.float32)
    tmp5_tmp = tl.zeros([$(1)], dtype=tl.float32)
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tl.store(Mean + x0, tmp3)
    tmp6 = $(rnumel)
    tmp7 = tmp4 / tl.toReal(tmp6)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.rsqrt(tmp9)
    tl.debug_barrier()
    tl.store(Rstd + x0, tmp10)
  } : ComputeKernel).surfaceBody).drop 4

def normalizeChunk (X W Bias Y : RegionName) (D B k : Nat) : List ComputeStmt :=
  [ ComputeStmt.assign .nat [] "xrow" (.alg (.remap (shape := [1,1]) []
      (fun _ => ⟨⟨0,by decide⟩,⟨0,by decide⟩,()⟩) (.ref .nat [1,1] "x0"))),
    ComputeStmt.assign .real [] "rowMean" (.alg (.remap (shape := [1,1]) []
      (fun _ => ⟨⟨0,by decide⟩,⟨0,by decide⟩,()⟩) (.ref .real [1,1] "tmp3"))),
    ComputeStmt.assign .real [] "rowInverse" (.alg (.remap (shape := [1,1]) []
      (fun _ => ⟨⟨0,by decide⟩,⟨0,by decide⟩,()⟩) (.ref .real [1,1] "tmp10"))) ] ++
  ((triton {
    tl.region X = tl.float32, W = tl.float32, Bias = tl.float32, Y = tl.bfloat16
    xrow = tl.program_id(0)
    rowMean = 0.0
    rowInverse = 0.0
    roffset = $(k * B)
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
  } : ComputeKernel).surfaceBody).drop 3

def normalizeCode (X W Bias Y : RegionName) (D B : Nat) : Nat → Nat → List ComputeStmt
  | _, 0 => []
  | start, count + 1 => normalizeChunk X W Bias Y D B start ++
      normalizeCode X W Bias Y D B (start + 1) count

/-- The three expressions are the selected ordered Welford helper tree.
They are bound to its source expansion in Proofs.Source, not numerical oracles. -/
def tupleCode (mean m2 weight : Op .real []) : List ComputeStmt :=
  [ .assign .real [1] "tmp3_tmp" (.compute (.alg .fp32 (.full [1] mean))),
    .assign .real [1] "tmp4_tmp" (.compute (.alg .fp32 (.full [1] m2))),
    .assign .real [1] "tmp5_tmp" (.compute (.alg .fp32 (.full [1] weight))) ]

def blockCount (D B : Nat) : Nat := (D + B - 1) / B

/-- Complete source row: both loops, ordered statistics stores and BF16 Y.
Natural indices are connected to the source's explicit int64 arithmetic by
Proofs.NativeIndex; no FP cast is erased here. -/
def fusedKernel (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (mean m2 weight : Op .real []) : ComputeKernel :=
  .mk [X,W,Bias] [Y,Mean,Rstd]
    ((welfordPrefix X S D B ++ chunkCode X D B 1 (blockCount D B - 1) ++
      tupleCode mean m2 weight ++ welfordSuffix Rstd Mean D) ++
      normalizeCode X W Bias Y D B 0 (blockCount D B))

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.LegacySource
