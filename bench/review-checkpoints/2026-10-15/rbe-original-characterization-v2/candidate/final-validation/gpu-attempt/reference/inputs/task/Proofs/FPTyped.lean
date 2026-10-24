import bench.tritonbench_g.rbe_triton_transform.RbeTritonTransform
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

/-! The existing complete rotary pair, with actual fp16 storage and fp32
frequency/arithmetic promotion. Implicit Python promotions and final storage
conversion are materialized. The native transcendental callbacks remain
explicit; this module alone is not a closed binary-value profile. -/
namespace VeriTile.Bench.TritonBenchG.RbeTritonTransform.FPTyped
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
structure Parameters where
  (input output : RegionName)
  (rows columns inputBatchStride inputRowStride inputColumnStride
    outputBatchStride outputRowStride outputColumnStride startToken : Nat)
  theta : ℝ
  (frequencyDimension blockRows blockColumns : Nat)

def typed (dtype : FloatDType)
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  tl.region x_ptr = DTYPE, out_ptr = DTYPE
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv($(K), $(BLOCK_SIZE_K))
  pid_n = pid % tl.cdiv($(K), $(BLOCK_SIZE_K))

  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K) // $(2)) * $(2)
  x_ptrs = x_ptr + (pid_batch * $(stride_x_batch) + $(stride_x_m) * offs_m[:, None] +
    $(stride_x_n) * offs_n[None, :])
  x_real_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(K))
  real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
  x_imag_mask = (offs_m[:, None] < $(M)) & ($((1 : Nat)) + offs_n[None, :] < $(K))
  imag = tl.load(x_ptrs + $(1), mask=x_imag_mask, other=0.0)
  tl.debug_barrier()
  start_block = $((start_token_position : Nat)) + pid_m * $(BLOCK_SIZE_M)
  freqs = offs_n % $(DIM)
  freqs_f = tl.toReal(freqs).to(tl.float32) / $(DIM)
  freqs_p = tl.extra.cuda.libdevice.pow($((THETA : ℝ)), freqs_f)
  tks = tl.arange(0, $(BLOCK_SIZE_M)) + start_block
  tks_f = tl.toReal(tks).to(tl.float32)
  freqs_mn = tks_f[:, None] / freqs_p[None, :]
  cos = tl.cos(freqs_mn)
  sin = tl.sin(freqs_mn)

  out_real = real.to(tl.float32) * cos - imag.to(tl.float32) * sin
  out_imag = real.to(tl.float32) * sin + imag.to(tl.float32) * cos
  tl.debug_barrier()
  out_ptrs = out_ptr + (pid_batch * $(stride_out_batch) + $(stride_out_m) * offs_m[:, None] +
    $(stride_out_n) * offs_n[None, :])
  out_real_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(K))
  tl.store(out_ptrs, out_real.to(DTYPE), mask=out_real_mask)
  out_imag_mask = (offs_m[:, None] < $(M)) & ($((1 : Nat)) + offs_n[None, :] < $(K))
  tl.store(out_ptrs + $(1), out_imag.to(DTYPE), mask=out_imag_mask)
}

def source (p : Parameters) : ComputeKernel :=
  typed .fp16 p.input p.output p.rows p.columns p.inputBatchStride p.inputRowStride
    p.inputColumnStride p.outputBatchStride p.outputRowStride p.outputColumnStride
    p.startToken p.theta p.frequencyDimension p.blockRows p.blockColumns

def legacySource (p : Parameters) : ComputeKernel :=
  rbe_triton_surface p.input p.output p.rows p.columns p.inputBatchStride p.inputRowStride
    p.inputColumnStride p.outputBatchStride p.outputRowStride p.outputColumnStride
    p.startToken p.theta p.frequencyDimension p.blockRows p.blockColumns

def execution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [source, typed, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod]
  rfl

def parts (p : Parameters) : Nat := (p.columns + p.blockColumns - 1) / p.blockColumns
abbrev Index (p : Parameters) := TileIndex [p.blockRows, p.blockColumns / 2]

def row {α : Type} (p : Parameters) (s : State α) (i : Index p) : Nat :=
  s.pids 1 / parts p * p.blockRows + i.1.val

def column {α : Type} (p : Parameters) (s : State α) (i : Index p) : Nat :=
  s.pids 1 % parts p * p.blockColumns + i.2.1.val * 2

def inputOffset {α : Type} (p : Parameters) (s : State α) (imag : Bool) (i : Index p) : Nat :=
  s.pids 0 * p.inputBatchStride + p.inputRowStride * row p s i +
    p.inputColumnStride * column p s i + if imag then 1 else 0

def outputOffset {α : Type} (p : Parameters) (s : State α) (imag : Bool) (i : Index p) : Nat :=
  s.pids 0 * p.outputBatchStride + p.outputRowStride * row p s i +
    p.outputColumnStride * column p s i + if imag then 1 else 0

def active {α : Type} (p : Parameters) (s : State α) (imag : Bool) (i : Index p) : Prop :=
  row p s i < p.rows ∧ (if imag then 1 + column p s i else column p s i) < p.columns
instance {α : Type} (p : Parameters) (s : State α) (imag : Bool) :
    DecidablePred (active p s imag) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- Native pow/sin/cos and conversion callbacks remain explicit. This is a
backend-value reference, never Real.sin/Real.cos or an exact-rounding premise. -/
def phase {α : Type} (M : Algebra α) (p : Parameters) (s : State α) (i : Index p) : α :=
  let exponent := M.binary (some .fp32) .real .div
    (M.cast (some .fp32) .real .real (M.fromNat (some .fp32) (column p s i % p.frequencyDimension)))
    (M.literal (some .fp32) .real (p.frequencyDimension : ℝ))
  let power := M.binary (some .fp32) .real .pow (M.literal (some .fp32) .real p.theta) exponent
  M.binary (some .fp32) .real .div
    (M.cast (some .fp32) .real .real
      (M.fromNat (some .fp32) (i.1.val + (p.startToken + s.pids 1 / parts p * p.blockRows)))) power

def component {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (imag : Bool) (i : Index p) : α :=
  M.cast (some .fp32) .fp16 .real
    (if active p s imag i then (s.mem p.input (inputOffset p s imag i)).read .fp16
     else M.cast none .real .fp16 (M.literal none .real 0.0))

def value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (imag : Bool) (i : Index p) : α :=
  let re := component M p s Bool.false i
  let im := component M p s Bool.true i
  let c := M.unary (some .fp32) .cos (phase M p s i)
  let sn := M.unary (some .fp32) .sin (phase M p s i)
  M.cast none .real .fp16
    (if imag then M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul re sn) (M.binary (some .fp32) .real .mul im c)
    else M.binary (some .fp32) .real .sub
      (M.binary (some .fp32) .real .mul re c) (M.binary (some .fp32) .real .mul im sn))

def halfScatter {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (imag : Bool) (initial : State α) : State α :=
  (TileShape.allIndices [p.blockRows, p.blockColumns / 2]).foldl (fun t i =>
    if active p s imag i then t.write p.output (outputOffset p s imag i)
      (.mk .fp16 (value M p s imag i)) else t) initial

def written {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  halfScatter M p s Bool.true (halfScatter M p s Bool.false s)

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    (execution M p s).val.mem = (written M p s).mem := by
  simp +contextual [execution, written, halfScatter, masked_scatter_memory, value, component, phase,
    active, inputOffset, outputOffset, row, column, parts, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, integralFloorDiv, integralMod,
    and_assoc, and_left_comm, and_comm]

#axiomsClean execution
#axiomsClean execution_memory
end
end VeriTile.Bench.TritonBenchG.RbeTritonTransform.FPTyped
