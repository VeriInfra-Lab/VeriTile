import VeriTile.Triton.DSL.NamedFloatDTypeCasts
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Support.Parameters

/-! Complete repaired Python source and its existing head-index candidate.
Pointer storage, dot dtype, scalar bindings and geometry are caller metadata.
All signed casts, literal masks, pointer updates, the residual branch and
explicit output conversion remain inside their original function bodies. -/
namespace VeriTile.Bench.Repairs.BmmChunkBwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

abbrev HostCall := TypedSource.HostCall
abbrev Tuning := TypedSource.Tuning
abbrev tuning := TypedSource.tuning

/-- Native errors are observable host outcomes. They are not numerical laws. -/
inductive SetupError where
  | invalidShape | invalidLayout | allocation | device | compilation
  | launchConfiguration | gridLimit | unsupportedDType
  deriving DecidableEq, Repr

/-- Logical object identity is distinct from the storage region: two views may
share storage while the supplied output object must still be returned. -/
structure TensorView where
  object : Nat
  region : RegionName
  offset : Nat
  strides : TypedSource.TensorStrides
  negative : Bool := Bool.false

structure DoutView where
  region : RegionName
  offset : Nat
  strides : TypedSource.DoutStrides
  negative : Bool := Bool.false

structure Caller where
  a : TensorView
  dout : DoutView
  residual : TensorView
  output : TensorView
  suppliedOutput : Bool

/-- Complete logical tensor coordinates, independent of launch tiles. -/
structure Coordinate where
  batch : Nat
  sequence : Nat
  group : Nat
  feature : Nat
  deriving DecidableEq

def valid (h : HostCall) (x : Coordinate) : Prop :=
  x.batch < h.batch ∧ x.sequence < h.seqlen ∧
  x.group < h.groupCount ∧ x.feature < h.K

def TensorView.address (v : TensorView) (x : Coordinate) : Nat :=
  v.offset + x.batch * v.strides.batch + x.sequence * v.strides.sequence +
    x.group * v.strides.head + x.feature * v.strides.feature

def DoutView.address (v : DoutView) (b c g k r : Nat) : Nat :=
  v.offset + b*v.strides.batch + c*v.strides.chunk + g*v.strides.head +
    k*v.strides.row + r*v.strides.column

/-- Actual finite allocation windows returned by preparation APIs. -/
structure AllocationWindow where
  region : RegionName
  base : Nat
  size : Nat

def freshCell (allocations : List AllocationWindow) (r : RegionName) (a : Nat) : Prop :=
  ∃ w ∈ allocations, r = w.region ∧ w.base ≤ a ∧ a < w.base+w.size

/-- Exact write footprint, including holes in a strided supplied view. -/
def outputCell (h : HostCall) (v : TensorView) (r : RegionName) (a : Nat) : Prop :=
  ∃ x, valid h x ∧ r = v.region ∧ a = v.address x

/-- A value-independent necessary output domain. Arbitrary conflicting logical
values cannot be stored in one internally overlapping output cell. Input views
may overlap and all caller roles may alias each other. No conservative stride
heuristic is substituted for this exact injectivity condition. -/
def OutputInjective (h : HostCall) (v : TensorView) : Prop :=
  ∀ x y, valid h x → valid h y → v.address x = v.address y → x = y

namespace Kernels
open TypedSource (Storage)

def callSignature (aStorage doutStorage : Storage) (dtype out_dtype : FloatDType) : CallSignature where
  pointers := [("a_ptr",match aStorage with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16),
    ("dout_ptr",match doutStorage with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16)]
  floatingPointers := [("res_ptr",dtype),("db_ptr",out_dtype)]
  scalars := [("seqlen",.integer), ("chunk_size",.integer), ("K",.integer), ("ngroups",.integer), ("stride_a_batch",.integer), ("stride_a_seqlen",.integer), ("stride_a_head",.integer), ("stride_ak",.integer), ("stride_dout_batch",.integer), ("stride_dout_chunk",.integer), ("stride_dout_head",.integer), ("stride_dout_csize_m",.integer), ("stride_dout_csize_n",.integer), ("stride_db_batch",.integer), ("stride_db_seqlen",.integer), ("stride_db_head",.integer), ("stride_db_k",.integer), ("stride_res_batch",.integer), ("stride_res_seqlen",.integer), ("stride_res_head",.integer), ("stride_res_k",.integer)]
  constexpr := ["dot_dtype","HAS_RESIDUAL","BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_CS"]
  readPorts := ["a_ptr","dout_ptr","res_ptr"]
  writePorts := ["db_ptr"]

def sourceKernel (aStorage doutStorage : Storage) (dtype out_dtype dot_dtype : FloatDType)
    (a_ptr dout_ptr db_ptr res_ptr : RegionName)
    (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m stride_dout_csize_n stride_db_batch stride_db_seqlen stride_db_head stride_db_k stride_res_batch stride_res_seqlen stride_res_head stride_res_k BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat) (HAS_RESIDUAL : Bool) : ComputeKernel :=
  with_triton_call (callSignature aStorage doutStorage dtype out_dtype) call_args [
    seqlen := seqlen, chunk_size := chunk_size, K := K, ngroups := ngroups, stride_a_batch := stride_a_batch, stride_a_seqlen := stride_a_seqlen, stride_a_head := stride_a_head, stride_ak := stride_ak, stride_dout_batch := stride_dout_batch, stride_dout_chunk := stride_dout_chunk, stride_dout_head := stride_dout_head, stride_dout_csize_m := stride_dout_csize_m, stride_dout_csize_n := stride_dout_csize_n, stride_db_batch := stride_db_batch, stride_db_seqlen := stride_db_seqlen, stride_db_head := stride_db_head, stride_db_k := stride_db_k, stride_res_batch := stride_res_batch, stride_res_seqlen := stride_res_seqlen, stride_res_head := stride_res_head, stride_res_k := stride_res_k] in triton {
    seqlen = tl.cast(seqlen, tl.int64)
    chunk_size = tl.cast(chunk_size, tl.int64)
    K = tl.cast(K, tl.int64)
    ngroups = tl.cast(ngroups, tl.int64)
    stride_a_batch = tl.cast(stride_a_batch, tl.int64)
    stride_a_seqlen = tl.cast(stride_a_seqlen, tl.int64)
    stride_a_head = tl.cast(stride_a_head, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_dout_batch = tl.cast(stride_dout_batch, tl.int64)
    stride_dout_chunk = tl.cast(stride_dout_chunk, tl.int64)
    stride_dout_head = tl.cast(stride_dout_head, tl.int64)
    stride_dout_csize_m = tl.cast(stride_dout_csize_m, tl.int64)
    stride_dout_csize_n = tl.cast(stride_dout_csize_n, tl.int64)
    stride_db_batch = tl.cast(stride_db_batch, tl.int64)
    stride_db_seqlen = tl.cast(stride_db_seqlen, tl.int64)
    stride_db_head = tl.cast(stride_db_head, tl.int64)
    stride_db_k = tl.cast(stride_db_k, tl.int64)
    stride_res_batch = tl.cast(stride_res_batch, tl.int64)
    stride_res_seqlen = tl.cast(stride_res_seqlen, tl.int64)
    stride_res_head = tl.cast(stride_res_head, tl.int64)
    stride_res_k = tl.cast(stride_res_k, tl.int64)
    pid_b = tl.program_id(axis=1).to(tl.int64)
    pid_ch = tl.program_id(axis=2).to(tl.int64)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0).to(tl.int64) // num_pid_n
    pid_n = tl.program_id(axis=0).to(tl.int64) % num_pid_n

    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * stride_dout_chunk + pid_h * stride_dout_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cs = tl.arange(0, BLOCK_SIZE_CS).to(tl.int64)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_csize_n + offs_cs[None, :] * stride_dout_csize_m)
    a_ptrs = a_ptr + (offs_cs[:, None] * stride_a_seqlen + offs_n[None, :] * stride_ak)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for cs in range(0, tl.cdiv(chunk_size_limit, BLOCK_SIZE_CS)):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_cs[None, :] < chunk_size_limit - cs * BLOCK_SIZE_CS), other=0.0).to(dot_dtype)
        a = tl.load(a_ptrs, mask=(offs_cs[:, None] < chunk_size_limit - cs * BLOCK_SIZE_CS) & (offs_n[None, :] < K), other=0.0).to(dot_dtype)
        acc += tl.dot(dout, a)
        dout_ptrs += BLOCK_SIZE_CS * stride_dout_csize_m
        a_ptrs += BLOCK_SIZE_CS * stride_a_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_RESIDUAL:
        res_ptr += pid_b * stride_res_batch + pid_c * chunk_size * stride_res_seqlen + pid_h * stride_res_head
        res_ptrs = res_ptr + (offs_m[:, None] * stride_res_seqlen + offs_n[None, :] * stride_res_k)
        res = tl.load(res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K)).to(tl.float32)
        acc += res
    db = acc.to(db_ptr.dtype.element_ty)

    db_ptr += pid_b * stride_db_batch + pid_c * chunk_size * stride_db_seqlen + pid_h * stride_db_head
    db_ptrs = db_ptr + (offs_m[:, None] * stride_db_seqlen + offs_n[None, :] * stride_db_k)
    tl.store(db_ptrs, db, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K))
  }

def candidateKernel (aStorage doutStorage : Storage) (dtype out_dtype dot_dtype : FloatDType)
    (a_ptr dout_ptr db_ptr res_ptr : RegionName)
    (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m stride_dout_csize_n stride_db_batch stride_db_seqlen stride_db_head stride_db_k stride_res_batch stride_res_seqlen stride_res_head stride_res_k BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat) (HAS_RESIDUAL : Bool) : ComputeKernel :=
  with_triton_call (callSignature aStorage doutStorage dtype out_dtype) call_args [
    seqlen := seqlen, chunk_size := chunk_size, K := K, ngroups := ngroups, stride_a_batch := stride_a_batch, stride_a_seqlen := stride_a_seqlen, stride_a_head := stride_a_head, stride_ak := stride_ak, stride_dout_batch := stride_dout_batch, stride_dout_chunk := stride_dout_chunk, stride_dout_head := stride_dout_head, stride_dout_csize_m := stride_dout_csize_m, stride_dout_csize_n := stride_dout_csize_n, stride_db_batch := stride_db_batch, stride_db_seqlen := stride_db_seqlen, stride_db_head := stride_db_head, stride_db_k := stride_db_k, stride_res_batch := stride_res_batch, stride_res_seqlen := stride_res_seqlen, stride_res_head := stride_res_head, stride_res_k := stride_res_k] in triton {
    seqlen = tl.cast(seqlen, tl.int64)
    chunk_size = tl.cast(chunk_size, tl.int64)
    K = tl.cast(K, tl.int64)
    ngroups = tl.cast(ngroups, tl.int64)
    stride_a_batch = tl.cast(stride_a_batch, tl.int64)
    stride_a_seqlen = tl.cast(stride_a_seqlen, tl.int64)
    stride_a_head = tl.cast(stride_a_head, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_dout_batch = tl.cast(stride_dout_batch, tl.int64)
    stride_dout_chunk = tl.cast(stride_dout_chunk, tl.int64)
    stride_dout_head = tl.cast(stride_dout_head, tl.int64)
    stride_dout_csize_m = tl.cast(stride_dout_csize_m, tl.int64)
    stride_dout_csize_n = tl.cast(stride_dout_csize_n, tl.int64)
    stride_db_batch = tl.cast(stride_db_batch, tl.int64)
    stride_db_seqlen = tl.cast(stride_db_seqlen, tl.int64)
    stride_db_head = tl.cast(stride_db_head, tl.int64)
    stride_db_k = tl.cast(stride_db_k, tl.int64)
    stride_res_batch = tl.cast(stride_res_batch, tl.int64)
    stride_res_seqlen = tl.cast(stride_res_seqlen, tl.int64)
    stride_res_head = tl.cast(stride_res_head, tl.int64)
    stride_res_k = tl.cast(stride_res_k, tl.int64)
    pid_b = tl.program_id(axis=1).to(tl.int64)
    pid_ch = tl.program_id(axis=2).to(tl.int64)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - ngroups * pid_c
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0).to(tl.int64) // num_pid_n
    pid_n = tl.program_id(axis=0).to(tl.int64) % num_pid_n

    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * stride_dout_chunk + pid_h * stride_dout_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cs = tl.arange(0, BLOCK_SIZE_CS).to(tl.int64)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_csize_n + offs_cs[None, :] * stride_dout_csize_m)
    a_ptrs = a_ptr + (offs_cs[:, None] * stride_a_seqlen + offs_n[None, :] * stride_ak)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for cs in range(0, tl.cdiv(chunk_size_limit, BLOCK_SIZE_CS)):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_cs[None, :] < chunk_size_limit - cs * BLOCK_SIZE_CS), other=0.0).to(dot_dtype)
        a = tl.load(a_ptrs, mask=(offs_cs[:, None] < chunk_size_limit - cs * BLOCK_SIZE_CS) & (offs_n[None, :] < K), other=0.0).to(dot_dtype)
        acc += tl.dot(dout, a)
        dout_ptrs += BLOCK_SIZE_CS * stride_dout_csize_m
        a_ptrs += BLOCK_SIZE_CS * stride_a_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_RESIDUAL:
        res_ptr += pid_b * stride_res_batch + pid_c * chunk_size * stride_res_seqlen + pid_h * stride_res_head
        res_ptrs = res_ptr + (offs_m[:, None] * stride_res_seqlen + offs_n[None, :] * stride_res_k)
        res = tl.load(res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K)).to(tl.float32)
        acc += res
    db = acc.to(db_ptr.dtype.element_ty)

    db_ptr += pid_b * stride_db_batch + pid_c * chunk_size * stride_db_seqlen + pid_h * stride_db_head
    db_ptrs = db_ptr + (offs_m[:, None] * stride_db_seqlen + offs_n[None, :] * stride_db_k)
    tl.store(db_ptrs, db, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K))
  }

def dotDType (a dout : Storage) : FloatDType :=
  match a,dout with
  | .bf16,_ | _,.bf16 => .bf16
  | .fp16,_ | _,.fp16 => .fp16
  | _,_ => .fp32

def family (aStorage doutStorage : Storage) (dtype out_dtype : FloatDType)
    (a_ptr dout_ptr db_ptr res_ptr : RegionName)
    (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m stride_dout_csize_n stride_db_batch stride_db_seqlen stride_db_head stride_db_k stride_res_batch stride_res_seqlen stride_res_head stride_res_k BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat) (HAS_RESIDUAL : Bool) : ComputeKernel :=
  sourceKernel aStorage doutStorage dtype out_dtype (dotDType aStorage doutStorage) a_ptr dout_ptr db_ptr res_ptr seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m stride_dout_csize_n stride_db_batch stride_db_seqlen stride_db_head stride_db_k stride_res_batch stride_res_seqlen stride_res_head stride_res_k BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS HAS_RESIDUAL

def candidateFamily (aStorage doutStorage : Storage) (dtype out_dtype : FloatDType)
    (a_ptr dout_ptr db_ptr res_ptr : RegionName)
    (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m stride_dout_csize_n stride_db_batch stride_db_seqlen stride_db_head stride_db_k stride_res_batch stride_res_seqlen stride_res_head stride_res_k BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat) (HAS_RESIDUAL : Bool) : ComputeKernel :=
  candidateKernel aStorage doutStorage dtype out_dtype (dotDType aStorage doutStorage) a_ptr dout_ptr db_ptr res_ptr seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m stride_dout_csize_n stride_db_batch stride_db_seqlen stride_db_head stride_db_k stride_res_batch stride_res_seqlen stride_res_head stride_res_k BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS HAS_RESIDUAL

def quotedOriginal := family .fp32 .fp32
def quotedIndexCommuted := candidateFamily .fp32 .fp32
end Kernels
end VeriTile.Bench.Repairs.BmmChunkBwd

namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.TypedSource
open VeriTile.Triton

/-- Every retained storage profile invokes the current repaired source. The
retired residual-order selector is kept only for internal proof compatibility;
it no longer introduces an additional kernel implementation. The current
index candidate has its explicit source entry in Kernels. -/
def specialize (a dout : Storage) (_retiredResidualOrder : Bool) :=
  VeriTile.Bench.Repairs.BmmChunkBwd.Kernels.family a dout

def HostCall.source (h : HostCall) (c : Tuning) (swapped : Bool) : ComputeKernel :=
  specialize h.aStorage h.doutStorage swapped h.residualDType h.outputDType
    h.A h.Dout h.Out h.Residual h.seqlen h.chunk h.K h.groupCount
    h.aStrides.batch h.aStrides.sequence (h.headStride h.aStrides.head) h.aStrides.feature
    h.doutStrides.batch h.doutStrides.chunk (h.headStride h.doutStrides.head) h.doutStrides.row h.doutStrides.column
    h.outStrides.batch h.outStrides.sequence (h.headStride h.outStrides.head) h.outStrides.feature
    (h.resStride h.residualStrides.batch) (h.resStride h.residualStrides.sequence)
    (h.resStride (h.headStride h.residualStrides.head)) (h.resStride h.residualStrides.feature)
    c.BM c.BN c.BCS h.hasResidual

end VeriTile.Bench.TritonBenchG.BmmChunkBwd.TypedSource

namespace VeriTile.Bench.Repairs.BmmChunkBwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
namespace Kernels
def bindMain (version : Bool) (h : HostCall) (c : Tuning) : ComputeKernel :=
  (if version then quotedIndexCommuted else quotedOriginal)
    h.residualDType h.outputDType h.A h.Dout h.Out h.Residual h.seqlen h.chunk h.K h.groupCount
    h.aStrides.batch h.aStrides.sequence (h.headStride h.aStrides.head) h.aStrides.feature
    h.doutStrides.batch h.doutStrides.chunk (h.headStride h.doutStrides.head) h.doutStrides.row h.doutStrides.column
    h.outStrides.batch h.outStrides.sequence (h.headStride h.outStrides.head) h.outStrides.feature
    (h.resStride h.residualStrides.batch) (h.resStride h.residualStrides.sequence)
    (h.resStride (h.headStride h.residualStrides.head)) (h.resStride h.residualStrides.feature)
    c.BM c.BN c.BCS h.hasResidual

def fp32Source := bindMain Bool.false
def indexCommuted := bindMain Bool.true

theorem fp32Source_eq (h : HostCall) (c : Tuning)
    (ha : h.aStorage=.fp32) (hd : h.doutStorage=.fp32) :
    fp32Source h c=h.source c Bool.false := by
  simp only [fp32Source,bindMain,quotedOriginal,TypedSource.HostCall.source,ha,hd,TypedSource.specialize,VeriTile.Bench.Repairs.BmmChunkBwd.Kernels.quotedOriginal,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody]

#axiomsClean fp32Source_eq
end Kernels

end VeriTile.Bench.Repairs.BmmChunkBwd
