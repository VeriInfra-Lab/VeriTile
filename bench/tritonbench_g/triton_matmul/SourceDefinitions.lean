import VeriTile.Triton
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
/-- Both actual host storage profiles, with full original grouped stream.
The three-argument dot remains dotAcc, never a separate dot-plus-add. -/
def typed (dtype : FloatDType)
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  start_m = pid_m * $(BLOCK_SIZE_M)
  start_n = pid_n * $(BLOCK_SIZE_N)
  offs_am = start_m + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = start_n + tl.arange(0, $(BLOCK_SIZE_N))
  offs_am = tl.where(offs_am < $(M), offs_am, $(0))
  offs_bn = tl.where(offs_bn < $(N), offs_bn, $(0))
  offs_am = tl.max_contiguous(tl.multiple_of(offs_am, $(BLOCK_SIZE_M)), $(BLOCK_SIZE_M))
  offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, $(BLOCK_SIZE_N)), $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for kk in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0, dtype=DTYPE)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0, dtype=DTYPE)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c = (accumulator).to(DTYPE)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}


inductive Storage where | half | e4m3
  deriving DecidableEq

def dtype : Storage → FloatDType | .half => .fp16 | .e4m3 => .f8e4
structure Parameters where
  (A B C : RegionName)
  (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat)
def source (p : Parameters) (st : Storage) : ComputeKernel :=
  typed (dtype st) p.A p.B p.C p.M p.N p.K p.stride_am p.stride_ak p.stride_bk p.stride_bn p.stride_cm p.stride_cn p.BLOCK_SIZE_M p.BLOCK_SIZE_N p.BLOCK_SIZE_K p.GROUP_SIZE_M p.numKBlocks
abbrev BM (p : Parameters) := p.BLOCK_SIZE_M
abbrev BN (p : Parameters) := p.BLOCK_SIZE_N
abbrev BK (p : Parameters) := p.BLOCK_SIZE_K
abbrev AIndex (p : Parameters) := TileIndex [BM p,BK p]
abbrev BIndex (p : Parameters) := TileIndex [BK p,BN p]
abbrev CIndex (p : Parameters) := TileIndex [BM p,BN p]
def prelude (p : Parameters) (st : Storage) := (source p st).surfaceBody.take 21
def body (p : Parameters) (st : Storage) : List ComputeStmt :=
  match (source p st).surfaceBody[21]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def epilogue (p : Parameters) (st : Storage) := (source p st).surfaceBody.drop 22


end
end VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
