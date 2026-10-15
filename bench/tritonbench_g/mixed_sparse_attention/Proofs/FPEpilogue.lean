import bench.tritonbench_g.mixed_sparse_attention.Proofs.FPMemory
namespace VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def epilogueSource (AccPre LPre : RegionName) (c : Config) : ComputeKernel :=
  let Seqlens := c.seq
  let Out := c.out
  triton {
    tl.region AccPre = tl.float32, LPre = tl.float32, Out = tl.float16
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    seqlen = tl.load(Seqlens + off_hz // $(c.H))
    offs_m = start_m * $(c.BM) + tl.arange(0, $(c.BM))
    offs_d = tl.arange(0, $(c.BD))
    qo_offset = (off_hz // $(c.H)) * $(c.sqz) + (off_hz % $(c.H)) * $(c.sqh)
    m_mask = offs_m[:, None] < seqlen
    acc = tl.load(AccPre + qo_offset + offs_m[:, None] * $(c.som) + offs_d[None, :] * $(c.sok))
    l_i = tl.load(LPre + offs_m)
    acc = acc / l_i[:, None]
    tl.store(Out + qo_offset + offs_m[:, None] * $(c.som) + offs_d[None, :] * $(c.sok),
      (acc).to(Out.dtype.element_ty), mask=m_mask)
  }

def epilogueInputs {α : Type} [Inhabited α] (M : Algebra α) (c : Config) (s : State α)
    (AccPre LPre : RegionName) : Accumulator α c.BM c.BD :=
  { maximum := fun _ => M.negInf
    normalizer := fun i => M.fp32Load ((s.mem LPre (rowStart c s+i.1.val)).read .real)
    output := fun i => M.fp32Load ((s.mem AccPre
      (queryBase c s+(rowStart c s+i.1.val)*c.som+i.2.1.val*c.sok)).read .real) }

/-- Original materialized accumulator/denominator epilogue boundary, with
fp32 scratch storage and fp16 output. There is no nonzero or positivity
premise on the denominator: the backend determines IEEE exceptional values. -/
specification epilogue_fp16 {α : Type} [Inhabited α] (M : Algebra α) (AccPre LPre : RegionName)
    (c : Config) (s : State α) :
    ∃ final, FP.Structural.exec M (epilogueSource AccPre LPre c) s = some final ∧
      final.mem = (outputMemory M c s (epilogueInputs M c s AccPre LPre)).mem := by
  simp [epilogueSource,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,store,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
    integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,epilogueInputs,outputMemory,
    rowStart,queryBase,sequenceLength]
  apply maskedMemory_congr
  rfl

#axiomsClean epilogue_fp16
end
end VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
