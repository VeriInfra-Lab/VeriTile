import bench.tritonbench_g.rbe_triton_transform.Proofs.Bridge
import bench.tritonbench_g.rbe_triton_transform.Proofs.Quotation

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Selected
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Selected FP32 Python kernel, with the frequency helper inlined. All integer
widenings, the sole explicit FP32 frequency cast, barriers and masks are kept.
The signed integer division implicitly converts its numerator to FP32. -/
def typed (dtype : FloatDType)
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      : Nat)
    (start_token_position : Int)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) :
    ComputeKernel := rbe_triton {
  tl.region x_ptr = DTYPE, out_ptr = DTYPE
  M = $((M : Nat)).to(tl.int64)
  K = $((K : Nat)).to(tl.int64)
  stride_x_batch = $((stride_x_batch : Nat)).to(tl.int64)
  stride_x_m = $((stride_x_m : Nat)).to(tl.int64)
  stride_x_n = $((stride_x_n : Nat)).to(tl.int64)
  stride_out_batch = $((stride_out_batch : Nat)).to(tl.int64)
  stride_out_m = $((stride_out_m : Nat)).to(tl.int64)
  stride_out_n = $((stride_out_n : Nat)).to(tl.int64)
  start_token_position = tl.cast($((start_token_position : Int)), tl.int64)
  pid_batch = tl.program_id(axis=0).to(tl.int64)
  pid = tl.program_id(axis=1).to(tl.int64)
  pid_m = pid // tl.cdiv(K, $((BLOCK_SIZE_K : Nat)))
  pid_n = pid % tl.cdiv(K, $((BLOCK_SIZE_K : Nat)))

  offs_m = pid_m * $((BLOCK_SIZE_M : Nat)) + tl.arange(0, $((BLOCK_SIZE_M : Nat))).to(tl.int64)
  offs_n = pid_n * $((BLOCK_SIZE_K : Nat)) + tl.arange(0, $((BLOCK_SIZE_K : Nat)) // $(2)).to(tl.int64) * $(2)
  x_ptrs = x_ptr + (pid_batch * stride_x_batch + stride_x_m * offs_m[:, None] +
    stride_x_n * offs_n[None, :])
  x_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
  real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
  x_imag_mask = (offs_m[:, None] < M) & ($((1 : Nat)) + offs_n[None, :] < K)
  imag = tl.load(x_ptrs + stride_x_n, mask=x_imag_mask, other=0.0)
  tl.debug_barrier()
  start_block = start_token_position + pid_m * $((BLOCK_SIZE_M : Nat))
  freqs = offs_n % $((DIM : Nat))
  freqs_f = tl.toReal(freqs).to(tl.float32) / $((DIM : Nat))
  freqs_p = tl.extra.cuda.libdevice.pow($((THETA : ℝ)), freqs_f)
  tks = tl.arange(0, $((BLOCK_SIZE_M : Nat))).to(tl.int64) + tl.cast(start_block, tl.int64)
  tks_f = tl.toReal(tks)
  freqs_mn = tks_f[:, None] / freqs_p[None, :]
  cos = tl.cos(freqs_mn)
  sin = tl.sin(freqs_mn)

  out_real = real * cos - imag * sin
  out_imag = real * sin + imag * cos
  tl.debug_barrier()
  out_ptrs = out_ptr + (pid_batch * stride_out_batch + stride_out_m * offs_m[:, None] +
    stride_out_n * offs_n[None, :])
  out_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
  tl.store(out_ptrs, out_real, mask=out_real_mask)
  out_imag_mask = (offs_m[:, None] < M) & ($((1 : Nat)) + offs_n[None, :] < K)
  tl.store(out_ptrs + stride_out_n, out_imag, mask=out_imag_mask)
}

def source (p : Source.Parameters) : ComputeKernel :=
  typed .fp32 p.input p.output p.rows p.columns p.inputBatchStride p.inputRowStride
    p.inputColumnStride p.outputBatchStride p.outputRowStride p.outputColumnStride
    p.startToken p.theta p.frequencyDimension p.blockRows p.blockColumns

def carrierTypes (ss : List ComputeStmt) : List (RegName × AlgDType) :=
  ss.filterMap fun statement =>
    let binding := match statement with
      | .alg (.assign dtype _ name _) => some (name,dtype)
      | .assign dtype _ name _ => some (name,dtype)
      | _ => none
    binding.filter fun entry => ["M","K","start_token_position","pid_batch","pid",
      "pid_m","pid_n","offs_m","offs_n","start_block","freqs","tks"].contains entry.1

theorem carrier_types (p : Source.Parameters) :
    carrierTypes (source p).surfaceBody=
      [("M",.nat),("K",.nat),("start_token_position",.int),("pid_batch",.nat),
       ("pid",.nat),("pid_m",.nat),("pid_n",.nat),("offs_m",.nat),("offs_n",.nat),
       ("start_block",.int),("freqs",.nat),("tks",.int)] := by rfl

def storeTypes (ss : List ComputeStmt) : List AlgDType :=
  ss.filterMap fun statement => match statement with
    | .store dtype _ _ _ _ => some dtype
    | _ => none

theorem store_types (p : Source.Parameters) :
    storeTypes (source p).surfaceBody=[.fp32,.fp32] := by rfl

def execution {α : Type} [Inhabited α] (M : Algebra α) (p : Source.Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [source,typed,FP.Structural.exec,ComputeKernel.surfaceBody,FP.Structural.run,
    FP.Structural.step,FP.Structural.evalExpr,FP.Structural.evalComputeOp,
    FP.Structural.evalOp_unfold,FP.Structural.evalOffsets,FP.Structural.store,
    FP.Structural.numeric,FP.Structural.bop,FP.Structural.natLt,FP.Structural.numericLt,FP.Structural.numericEq,
    ComputeDType.eraseDType,FP.Structural.ofFloat,FP.Structural.toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Selected
