import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeHost

/-! The two-pass source, specialized at XBLOCK=1 and expanded over its finite
range in increasing chunk order. Alpha-renamed accumulator/statistics registers
permit reuse of the identical typed BF16 normalization/store proof. The explicit
binary sum tree is a structural lowering profile, never a whole-result premise. -/
namespace VeriTile.Bench.Repairs.LayerNormWelfold
open VeriTile.Triton FP.Structural
namespace Reuse
export VeriTile.Bench.TritonBenchG.FusedLayernormTriton.FPStages
  (ReductionTree withEquality)
export VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
  (blockCount block_count_covers block_count_positive column_chunk chunk_unique
   row_column_injective output_bounds statistics_reinterpret normalizeCode
   normalizeValue normalize_whole_row NormalizeRegisters Separation rowMiss
   atRow runRows Tensor Allocations StorageDType InputContract Returned
   emptyStrided reinterpret outputTensor inverseBacking inverseTensor)
end Reuse
open Reuse
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def init (S B : Nat) : List ComputeStmt := (triton {
  xoffset = tl.program_id(0) * $(1)
  xindex = xoffset + tl.arange(0, $(1))[:, None]
  xmask = xindex < $(S)
  x0 = xindex
  tmp3 = tl.full([$(1), $(1)], 0, dtype=tl.float32)
  acc = tl.full([$(1), $(B)], 0, dtype=tl.float32)
} : ComputeKernel).surfaceBody

def reset (B : Nat) : List ComputeStmt := (triton {
  acc = tl.full([$(1), $(B)], 0, dtype=tl.float32)
} : ComputeKernel).surfaceBody

def chunk (variance : Bool) (X : RegionName) (D B k : Nat) : List ComputeStmt :=
  if variance then ((triton {
    tl.region X = tl.float32
    x0 = tl.program_id(0) + tl.arange(0, $(1))[:, None]
    acc = tl.full([$(1), $(B)], 0, dtype=tl.float32)
    tmp3 = tl.full([$(1), $(1)], 0, dtype=tl.float32)
    roffset = $(k * B)
    rindex = roffset + tl.arange(0, $(B))[None, :]
    rmask = rindex < $(D)
    x = tl.load(X + (rindex + ($(D) * x0)), mask=rmask, other=0.0).to(tl.float32)
    xx = (x).to(tl.float32)
    delta = xx - tmp3
    square = delta * delta
    guarded = tl.where(rmask, square, 0.0)
    broadcasted, _ = tl.broadcast(guarded, acc)
    next = acc + broadcasted
    acc = next
  } : ComputeKernel).surfaceBody).drop 3
  else ((triton {
    tl.region X = tl.float32
    x0 = tl.program_id(0) + tl.arange(0, $(1))[:, None]
    acc = tl.full([$(1), $(B)], 0, dtype=tl.float32)
    roffset = $(k * B)
    rindex = roffset + tl.arange(0, $(B))[None, :]
    rmask = rindex < $(D)
    x = tl.load(X + (rindex + ($(D) * x0)), mask=rmask, other=0.0).to(tl.float32)
    xx = (x).to(tl.float32)
    broadcasted, _ = tl.broadcast(xx, acc)
    next = acc + broadcasted
    acc = next
  } : ComputeKernel).surfaceBody).drop 2

def chunks (variance : Bool) (X : RegionName) (D B : Nat) : Nat → Nat → List ComputeStmt
  | _, 0 => []
  | start, count+1 => chunk variance X D B start ++ chunks variance X D B (start+1) count

def sumExpr {B : Nat} : ReductionTree B → Op .real []
  | .leaf i => .remap (shape := [1,B]) [] (fun _ => ⟨⟨0,by decide⟩,i,()⟩) (.ref .real [1,B] "acc")
  | .node l r => .add .real .nil (sumExpr l) (sumExpr r)

def sumCode {B : Nat} (tree : ReductionTree B) : List ComputeStmt :=
  [ .assign .real [1,1] "total" (.compute (.alg .fp32 (.full [1,1] (sumExpr tree)))) ]

/-- Mean is kept in-register through pass two and normalization, as in Python. -/
def meanCode (Mean : RegionName) (D : Nat) : List ComputeStmt :=
  ((triton {
    tl.region Mean = tl.float32
    total = tl.full([$(1), $(1)], 0, dtype=tl.float32)
    x0 = tl.program_id(0) + tl.arange(0, $(1))[:, None]
    denominator = $(D)
    mu = total / denominator
    tl.debug_barrier()
    tl.store(Mean + x0, mu)
  } : ComputeKernel).surfaceBody).drop 2 ++
  [ .assign .real [1,1] "tmp3" (.alg (.ref .real [1,1] "mu")) ]

def inverseCode (Rstd : RegionName) (D : Nat) : List ComputeStmt :=
  ((triton {
    tl.region Rstd = tl.float32
    total = tl.full([$(1), $(1)], 0, dtype=tl.float32)
    x0 = tl.program_id(0) + tl.arange(0, $(1))[:, None]
    denominator = $(D)
    variance = total / denominator
    epsilon = 1e-05
    inverse = tl.rsqrt(variance + epsilon)
    tl.debug_barrier()
    tl.store(Rstd + x0, inverse)
  } : ComputeKernel).surfaceBody).drop 2 ++
  [ .assign .real [1,1] "tmp10" (.alg (.ref .real [1,1] "inverse")) ]

end
end VeriTile.Bench.Repairs.LayerNormWelfold
