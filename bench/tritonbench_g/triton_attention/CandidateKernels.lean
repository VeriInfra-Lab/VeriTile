import bench.tritonbench_g.triton_attention.Kernels

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.CandidateSource
open VeriTile Triton FP.Structural
noncomputable section

def preprocessKernel (c : Shape) : ComputeKernel :=
  let dtype : FloatDType := .fp32
  let out_dtype : FloatDType := .fp32
  let Out := "Out"
  let DO := "DO"
  let L := "L"
  let NewDO := "NewDO"
  let Delta := "Delta"
  let BLOCK_M := 128
  let D_HEAD := c.dim
  triton {
  tl.region Out = DTYPE, DO = DTYPE, NewDO = DTYPE, L = OUT_DTYPE, Delta = OUT_DTYPE
  off_m = (tl.program_id(0) << $(7)) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(D_HEAD))
  o = (tl.load(Out + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  do_val = (tl.load(DO + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  denom = (tl.load(L + off_m)).to(tl.float32)
  do_val = do_val / denom[:, None]
  delta = tl.sum(o * do_val, axis=1)
  tl.store(NewDO + off_m[:, None] * $(D_HEAD) + off_n[None, :], do_val.to(DTYPE))
  tl.store(Delta + off_m, delta.to(OUT_DTYPE))
}

end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.CandidateSource
