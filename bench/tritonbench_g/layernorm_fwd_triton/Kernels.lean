import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.DSL.StaticSignedStrides
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Triton

/-- Input storage and all runtime scalars come from the call. The original
signed constructor, casts, registers and loops remain in the pasted body. -/
def callSignature (dtype : FloatDType) : CallSignature where
  pointers := []
  floatingPointers := [("X",dtype),("W",dtype),("Y",dtype)]
  scalars := [("stride_x_N",.integer),("stride_x_hn",.integer),("stride_x_hd",.integer),
    ("stride_y_N",.integer),("stride_y_hn",.integer),("stride_y_hd",.integer),
    ("stride_w_hn",.integer),("stride_w_hd",.integer),("N",.integer),("eps",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["X","W"]
  writePorts := ["Y"]

def runtimeKernel (dtype : FloatDType) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd stride_y_N stride_y_hn stride_y_hd
      stride_w_hn stride_w_hd N BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call (callSignature dtype) call_args
    [stride_x_N := stride_x_N, stride_x_hn := stride_x_hn, stride_x_hd := stride_x_hd,
     stride_y_N := stride_y_N, stride_y_hn := stride_y_hn, stride_y_hd := stride_y_hd,
     stride_w_hn := stride_w_hn, stride_w_hd := stride_w_hd, N := N] static_strides in triton {
    Seq = tl.program_id(0).to(tl.int64)
    H = tl.program_id(1).to(tl.int64)
    stride_x_N = tl.cast(stride_x_N, tl.int64)
    stride_x_hn = tl.cast(stride_x_hn, tl.int64)
    stride_x_hd = tl.cast(stride_x_hd, tl.int64)
    stride_y_N = tl.cast(stride_y_N, tl.int64)
    stride_y_hn = tl.cast(stride_y_hn, tl.int64)
    stride_y_hd = tl.cast(stride_y_hd, tl.int64)
    stride_w_hn = tl.cast(stride_w_hn, tl.int64)
    stride_w_hd = tl.cast(stride_w_hd, tl.int64)
    N = tl.cast(N, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), BLOCK_SIZE, tl.int64)
    lanes = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    X += Seq * stride_x_N + H * stride_x_hn
    Y += Seq * stride_y_N + H * stride_y_hn
    W += H * stride_w_hn

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(zero, N, step):
        cols = off + lanes
        a = tl.load(X + cols * stride_x_hd, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(zero, N, step):
        cols = off + lanes
        x = tl.load(X + cols * stride_x_hd, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    for off in range(zero, N, step):
        cols = off + lanes
        mask = cols < N
        w = tl.load(W + cols * stride_w_hd, mask=mask).to(tl.float32)
        x = tl.load(X + cols * stride_x_hd, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w
        tl.store(Y + cols * stride_y_hd, y.to(X.dtype.element_ty), mask=mask)
  }

def runtimeSourceCandidateKernel (dtype : FloatDType) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd stride_y_N stride_y_hn stride_y_hd
      stride_w_hn stride_w_hd N BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call (callSignature dtype) call_args
    [stride_x_N := stride_x_N, stride_x_hn := stride_x_hn, stride_x_hd := stride_x_hd,
     stride_y_N := stride_y_N, stride_y_hn := stride_y_hn, stride_y_hd := stride_y_hd,
     stride_w_hn := stride_w_hn, stride_w_hd := stride_w_hd, N := N] static_strides in triton {
    Seq = tl.program_id(0).to(tl.int64)
    H = tl.program_id(1).to(tl.int64)
    stride_x_N = tl.cast(stride_x_N, tl.int64)
    stride_x_hn = tl.cast(stride_x_hn, tl.int64)
    stride_x_hd = tl.cast(stride_x_hd, tl.int64)
    stride_y_N = tl.cast(stride_y_N, tl.int64)
    stride_y_hn = tl.cast(stride_y_hn, tl.int64)
    stride_y_hd = tl.cast(stride_y_hd, tl.int64)
    stride_w_hn = tl.cast(stride_w_hn, tl.int64)
    stride_w_hd = tl.cast(stride_w_hd, tl.int64)
    N = tl.cast(N, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), BLOCK_SIZE, tl.int64)
    lanes = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    X += Seq * stride_x_N + H * stride_x_hn
    Y += Seq * stride_y_N + H * stride_y_hn
    W += H * stride_w_hn

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(zero, N, step):
        cols = off + lanes
        a = tl.load(X + cols * stride_x_hd, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(zero, N, step):
        cols = off + lanes
        x = tl.load(X + cols * stride_x_hd, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    for off in range(zero, N, step):
        cols = off + lanes
        mask = cols < N
        x = tl.load(X + cols * stride_x_hd, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        w = tl.load(W + cols * stride_w_hd, mask=mask).to(tl.float32)
        y = x_hat * w
        tl.store(Y + cols * stride_y_hd, y.to(X.dtype.element_ty), mask=mask)
  }

/-- Immutable source pin; no original-source equivalence is claimed. -/
def originalSHA256 := "9ad4b038ee7d5fff373705d8ef8c0a4a9a3100a064db07703f89d61b32de6c88"
def candidateSHA256 := "a22618b08165692510fbf836cd2f7ad87943610902d3ee85016824e34e82aae9"
def block : Nat := 128

structure Shape where
  (sequences heads columns : Nat)
  deriving DecidableEq

structure Strides where
  (xSeq xHead xDim ySeq yHead yDim wHead wDim : Nat)
  deriving DecidableEq

namespace Source
open FP.Structural

structure Parameters where
  (x weight y : RegionName)
  (xSeqStride xHeadStride xDimStride ySeqStride yHeadStride yDimStride
    weightHeadStride weightDimStride columns block : Nat)

def kernel (p : Parameters) := runtimeKernel .fp32 p.x p.weight p.y
  p.xSeqStride p.xHeadStride p.xDimStride p.ySeqStride p.yHeadStride p.yDimStride
  p.weightHeadStride p.weightDimStride p.columns p.block

def parameters (s : Strides) (N : Nat) : Parameters :=
  ⟨"X", "W", "Y", s.xSeq, s.xHead, s.xDim, s.ySeq, s.yHead, s.yDim,
    s.wHead, s.wDim, N, 128⟩

def candidateKernel (p : Parameters) := runtimeSourceCandidateKernel .fp32 p.x p.weight p.y
  p.xSeqStride p.xHeadStride p.xDimStride p.ySeqStride p.yHeadStride p.yDimStride
  p.weightHeadStride p.weightDimStride p.columns p.block

def programs (shape : Shape) : List (Fin shape.sequences × Fin shape.heads) :=
  (List.finRange shape.sequences).flatMap fun s => (List.finRange shape.heads).map fun h => (s,h)

def atRow {α : Type} {shape : Shape} (s : State α) (row : Fin shape.sequences × Fin shape.heads) : State α :=
  {s with pids := fun dim => if dim=0 then row.1.val else if dim=1 then row.2.val else 0}

/-- Canonical logical kernel grid. Allocation, compilation, streams and host
return are outside this kernel entry; every program executes the typed source. -/
noncomputable def runRows {α : Type} [Inhabited α] {shape : Shape} (M : Algebra α) (p : Parameters) (epsilon : α) :
    List (Fin shape.sequences × Fin shape.heads) → State α → Option (State α)
  | [], s => some s
  | row :: rows, s => do
      let t ← FP.Structural.exec M (kernel p) (atRow (s.setReg "eps" .real [] (fun _ => epsilon)) row)
      runRows M p epsilon rows t

noncomputable def launch {α : Type} [Inhabited α] (M : Algebra α) (shape : Shape)
    (strides : Strides) (eps : α) (s : State α) : Option (State α) :=
  runRows M (parameters strides shape.columns) eps (programs shape) s

end Source

end VeriTile.Bench.Repairs.LayernormFwdTriton
