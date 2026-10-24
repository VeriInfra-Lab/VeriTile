import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
inductive Graph where | source | accumulator deriving DecidableEq, Repr
structure Parameters where
  (input weight output : RegionName)
  (rows columns inner inputRowStride inputInnerStride weightInnerStride weightColumnStride
    outputRowStride outputColumnStride blockRows blockColumns blockInner : Nat)
def typed (dtype : FloatDType)
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  tl.region A = DTYPE, B = DTYPE, C = DTYPE
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptr = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptr = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a_ptrs = a_ptr + k * $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs = b_ptr + k * $(BLOCK_SIZE_K) * $(stride_bk)
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    accumulator += tl.dot(a, b)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c.to(DTYPE), mask=c_mask)
}

def fusedTyped (dtype : FloatDType)
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  tl.region A = DTYPE, B = DTYPE, C = DTYPE
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptr = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptr = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a_ptrs = a_ptr + k * $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs = b_ptr + k * $(BLOCK_SIZE_K) * $(stride_bk)
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c.to(DTYPE), mask=c_mask)
}

def source (g : Graph) (p : Parameters) : ComputeKernel :=
  (match g with | .source => typed | .accumulator => fusedTyped) .fp32
    p.input p.weight p.output p.rows p.columns p.inner p.inputRowStride p.inputInnerStride
    p.weightInnerStride p.weightColumnStride p.outputRowStride p.outputColumnStride
    p.blockRows p.blockColumns p.blockInner

def prologue (p : Parameters) := (source .source p).surfaceBody.take 10
def body (g : Graph) (p : Parameters) : List ComputeStmt :=
  let code : List ComputeStmt := (source g p).surfaceBody
  match code[10]? with | some (ComputeStmt.forRangeDyn _ _ _ _ b) => b | _ => []
def suffix (p : Parameters) := (source .source p).surfaceBody.drop 11

def stopOp (p : Parameters) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat p.inner) (.constNat p.blockInner))
    (.constNat 1)) (.constNat p.blockInner)
theorem source_split (g : Graph) (p : Parameters) :
    (source g p).surfaceBody = prologue p ++
      [.forRangeDyn "k" (.constNat 0) (stopOp p) (.constNat 1) (body g p)] ++ suffix p := by
  cases g <;> rfl
#axiomsClean source_split
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
