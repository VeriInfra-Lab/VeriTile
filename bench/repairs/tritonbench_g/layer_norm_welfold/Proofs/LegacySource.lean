/- Unrolled proof reference. Its scalar ports follow the actual source:
libdevice.rsqrt and an assigned FP32 epsilon tensor. This is not the historical
Python body or a claim of FP equality with the former intrinsic port. The
canonical original/candidate quotations are in ../Kernels.lean; their complete
execution must be connected to this reference by ReadableBoundary. -/
import VeriTile.Triton
import VeriTile.Triton.Float.Structural

/-! Repaired Python source specialized at XBLOCK=1. The two finite statistics
passes retain every FP32 cast and masked-square operation; normalization retains
the BF16 store. A binary sum tree represents only the selected reduction order. -/
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Source
open VeriTile.Triton FP.Structural
noncomputable section

inductive ReductionTree (N : Nat) where
  | leaf : Fin N → ReductionTree N
  | node : ReductionTree N → ReductionTree N → ReductionTree N

def blockCount (D B : Nat) : Nat := (D + B - 1) / B

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
    epsilon = tl.full([], 1e-05, dtype=tl.float32)
    inverse = libdevice.rsqrt(variance + epsilon)
    tl.debug_barrier()
    tl.store(Rstd + x0, inverse)
  } : ComputeKernel).surfaceBody).drop 2 ++
  [ .assign .real [1,1] "tmp10" (.alg (.ref .real [1,1] "inverse")) ]

def statisticsKernel (X Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) : ComputeKernel :=
  .mk [X] [Mean,Rstd] (init S B ++ chunks Bool.false X D B 0 (blockCount D B) ++
    sumCode tree ++ meanCode Mean D ++ reset B ++
    chunks Bool.true X D B 0 (blockCount D B) ++ sumCode tree ++ inverseCode Rstd D)

def rowKernel (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) : ComputeKernel :=
  .mk [X,W,Bias] [Y,Mean,Rstd] ((statisticsKernel X Mean Rstd S D B tree).surfaceBody ++
    normalizeCode X W Bias Y D B 0 (blockCount D B))


/-- Kernel grid only: one program for each valid row. No allocation, device
selection, stream, autotune execution, or wrapper return is part of this run. -/
def runRows {α : Type} [Inhabited α] (M : Algebra α) (kernel : ComputeKernel) :
    List Nat → State α → Option (State α)
  | [], s => some s
  | row :: rows, s => do
      let t ← FP.Structural.exec M kernel {s with pids := fun dim => if dim = 0 then row else 0}
      runRows M kernel rows t

def launch {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) : Option (State α) :=
  runRows M
    (rowKernel "X" "W" "Bias" Y Mean Rstd S D B tree) (List.range S) s

end
end VeriTile.Bench.Repairs.LayerNormWelfold.Source
