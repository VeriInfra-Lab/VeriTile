import bench.tritonbench_g.quantize_global.Sources.Mathematical_Kernels

/-! Source definitions shared by the independent mathematical and floating
proofs. Production input precision is unspecified, so the task selects FP32;
the output allocation explicitly uses int8. Duplicate autotune entries and the
source's pid % group_size schedule are preserved. -/
namespace VeriTile.Bench.TritonBenchG.QuantTransposeKernel
open VeriTile.Triton
/-- Real-valued surface of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

This preserves the grouped one-dimensional program-id schedule, masked load,
global scale, CUDA `llrint` surface operation, transposed store addressing, and
masked writeback. The algorithm carrier records the pre-cast real value. -/
def quantize_global_transpose_real_surface
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  grid_m = ($((M : Nat)) + $((BLOCK_M : Nat)) - $((1 : Nat))) // $((BLOCK_M : Nat))
  grid_n = ($((N : Nat)) + $((BLOCK_N : Nat)) - $((1 : Nat))) // $((BLOCK_N : Nat))
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // group_size
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  A = A + (rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  a = tl.load(A, mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  B = B + (rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))
  tl.store(B, output, mask=mask)
}

/-- Proof-oriented scaled-store tile slice of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

The full Triton kernel uses a one-dimensional grouped program-id schedule to
derive `pid_m` and `pid_n`. This slice starts after that scheduling choice, uses
program axes 0/1 for the tile coordinates, loads the `BLOCK_M × BLOCK_N` tile
from `A`, applies the global `absmax_inv` scale, and proves the masked writeback
into `B`. CUDA `llrint` and int8 casting are outside VeriTile's current real-tile
arithmetic layer, matching the other quantization ports. -/
def quantize_global_transpose_scaled_store_slice
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  mask = (rm[:, None] < $(M)) & (rn[None, :] < $(N))
  a = tl.load(A + rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an),
    mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  output = $(scale127) * (a * absmax_inv)
  tl.store(B + rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn),
    output, mask=mask)
}

end VeriTile.Bench.TritonBenchG.QuantTransposeKernel

namespace VeriTile.Bench.TritonBenchG.QuantTransposeKernel
open VeriTile.Triton
/-- The already accepted pre-llrint scaled-store stage, with actual fp32 storage
and compute precision. This does not implement the final integer quantization. -/
def TypedScaled.source (A AbsmaxInv B : RegionName)
    (sam san sbn sbm M N BM BN : Nat) : ComputeKernel := triton {
  tl.region A = tl.float32, AbsmaxInv = tl.float32, B = tl.float32
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  rm = pid_m * $(BM) + tl.arange(0, $(BM))
  rn = pid_n * $(BN) + tl.arange(0, $(BN))
  mask = (rm[:, None] < $(M)) & (rn[None, :] < $(N))
  a = tl.load(A + rm[:, None] * $(sam) + rn[None, :] * $(san), mask=mask).to(tl.float32)
  absmax_inv = tl.load(AbsmaxInv).to(tl.float32)
  output = 127.0 * (a * absmax_inv)
  tl.store(B + rm[:, None] * $(sbm) + rn[None, :] * $(sbn), output, mask=mask)
}

end VeriTile.Bench.TritonBenchG.QuantTransposeKernel

namespace QuantizeIntegerStage
open VeriTile.Triton
/-- Complete quantization/store stage at supplied tile coordinates, with fp32
input/scale and int8 output. Grouped pid mapping and the host/grid wrapper
remain separate obligations. The source load, multiply order, llrint and
transposed masked store are executed. -/
def Transpose.source (A AbsmaxInv B : RegionName)
    (sam san sbn sbm M N BM BN : Nat) : ComputeKernel := triton {
  tl.region A = tl.float32, AbsmaxInv = tl.float32, B = tl.int8
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  rm = pid_m * $(BM) + tl.arange(0, $(BM))
  rn = pid_n * $(BN) + tl.arange(0, $(BN))
  mask = (rm[:, None] < $(M)) & (rn[None, :] < $(N))
  a = tl.load(A + rm[:, None] * $(sam) + rn[None, :] * $(san), mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))
  tl.store(B + rm[:, None] * $(sbm) + rn[None, :] * $(sbn), output, mask=mask)
}

end QuantizeIntegerStage

/-! Dtype-independent source coordinates and wrapper metadata. The historical
namespace is retained to preserve the accepted public statements. -/
namespace QuantTransposeFinite
def gridM (m : Nat) := (m+127)/128
def gridN (n : Nat) := (n+127)/128

def groupWidth (gn : Nat) := 8*gn
def groupId (gn pid : Nat) := pid/groupWidth gn
def groupSize (gm gn pid : Nat) := min (gm-groupId gn pid*8) 8
def tileRow (gm gn pid : Nat) := groupId gn pid*8+pid%groupSize gm gn pid
def tileCol (gm gn pid : Nat) := (pid%groupWidth gn)/groupSize gm gn pid

/-- Source pid uses pid%group_size, not residual%group_size. The final
partial group may therefore rotate the row order. -/
def inversePid (gm gn r c : Nat) : Nat :=
  let g := r/8
  let s := min (gm-g*8) 8
  let a := g*(8*gn)
  let rotation := a%s
  let u := if rotation≤r%8 then r%8-rotation else s+r%8-rotation
  a+c*s+u

def row (m n pid lane : Nat) := tileRow (gridM m) (gridN n) pid*128+lane/128
def col (m n pid lane : Nat) := tileCol (gridM m) (gridN n) pid*128+lane%128

/-- Two distinct decorator entries with identical actual launch metadata. -/
inductive Config where | first | second
  deriving DecidableEq, Repr

def Config.blockM (_ : Config) : Nat := 128
def Config.blockN (_ : Config) : Nat := 128
def Config.groupM (_ : Config) : Nat := 8
def Config.warps (_ : Config) : Nat := 4
def Config.stages (_ : Config) : Nat := 3

structure Parameters where
  M : Nat
  N : Nat
  sam : Nat
  san : Nat
  x : String
  scale : String
  output : String

def Parameters.n (p : Parameters) := p.M*p.N
def programs (p : Parameters) (_ : Config) : Nat := gridM p.M*gridN p.N
def offset (p : Parameters) (pid lane : Nat) := col p.M p.N pid lane*p.M+row p.M p.N pid lane
def inputOffset (p : Parameters) (i : Nat) := (i%p.M)*p.sam+(i/p.M)*p.san
def laneInput (p : Parameters) (pid lane : Nat) := row p.M p.N pid lane*p.sam+col p.M p.N pid lane*p.san
def active (p : Parameters) (pid lane : Nat) : Prop := row p.M p.N pid lane<p.M ∧ col p.M p.N pid lane<p.N
instance (p : Parameters) (pid lane : Nat) : Decidable (active p pid lane) := inferInstanceAs (Decidable (_ ∧ _))

def laneIndices (p : Parameters) (pid : Nat) (lanes : List Nat) :=
  lanes.filterMap (fun lane => if active p pid lane then some (offset p pid lane) else none)

def footprint (p : Parameters) (pids : List Nat) : List Nat :=
  pids.flatMap (fun pid => laneIndices p pid (List.range (128*128)))

structure Input where
  shape : List Nat
  strides : List Nat
  /-- Relative to the actual x.data_ptr(), including view stride effects. -/
  logicalOffsets : List Nat
  isCuda : Bool
  deviceIndex : Nat
  storageBase : Nat
  storageOffset : Nat
  storageLength : Nat

def Input.numel (a : Input) := a.shape.prod
def Input.M (a : Input) := a.shape.headD 0
def Input.N (a : Input) := a.shape[1]?.getD 0
def Input.sam (a : Input) := a.strides.headD 0
def Input.san (a : Input) := a.strides[1]?.getD 0

/-- Ordinary Torch nonnegative shape/stride metadata and tensor numel fit
signed64. This primitive metadata contract permits every runtime i32/i64
specialization and imposes no arithmetic or active-address safety condition. -/
def MetadataDomain (a : Input) : Prop :=
  a.M<2^63 ∧ a.N<2^63 ∧ a.sam<2^63 ∧ a.san<2^63 ∧ a.numel<2^63

def Input.outputShape (a : Input) := [a.N,a.M]


def viewOffsets : List Nat → List Nat → List Nat
  | [],[] => [0]
  | n::ns,t::ts => (List.range n).flatMap (fun i => (viewOffsets ns ts).map (fun o => i*t+o))
  | _,_ => []

structure Allocations where
  maximum : String
  scale : String
  output : String
  /-- Fresh torch.empty bytes need not be zero. -/
  initialOutput : Nat → BitVec 8
  maximumBase : Nat
  scaleBase : Nat
  outputBase : Nat
  currentDevice : Nat

def Allocations.Fresh (a : Allocations) (x : String) : Prop :=
  x≠a.maximum ∧ x≠a.scale ∧ x≠a.output ∧
  a.maximum≠a.scale ∧ a.maximum≠a.output ∧ a.scale≠a.output

def parameters (a : Input) (alloc : Allocations) (x : String) : Parameters :=
  ⟨a.M,a.N,a.sam,a.san,x,alloc.scale,alloc.output⟩

inductive Stage where
  | abs | maximum | unsqueeze | reciprocal | shapeUnpack | outputAllocation | outputShapeAssertion | inputStrideAssertion | outputStrideAssertion | signature | compilation | driverContext | gridParse | launchEnterHook | pointerPacking | launch | launchExitHook
  deriving DecidableEq, Repr

/-- Primitive exceptions are an explicit trusted runtime outcome, distinct
from source diagnostics. The no-failure profile is stated when proving success. -/
abbrev Environment := Stage → Option String

end QuantTransposeFinite
