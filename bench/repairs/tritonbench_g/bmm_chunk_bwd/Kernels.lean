import bench.repairs.tritonbench_g.bmm_chunk_bwd.Source

/-! Separate repair candidate. Floating source, masks, contraction order and all
nine configurations are reused unchanged. Python widens index operands before arithmetic and explicitly validates direct
FP32 caller buffers. A supplied output is written and returned directly. The integer-width
bridge and host boundary are separate from this mathematical typed source. -/
namespace VeriTile.Bench.Repairs.BmmChunkBwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd

abbrev HostCall := TypedSource.HostCall
abbrev Tuning := TypedSource.Tuning
abbrev tuning := TypedSource.tuning

def kernel (h : HostCall) (choice : Fin 9) : ComputeKernel :=
  h.source (tuning choice) Bool.false

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

/-- Complete ordinary FP32 quotations. Only the existing integer head-index
product changes in the candidate; floating operations and all flags remain. -/

def quotedOriginal :=
  fun (dtype out_dtype : FloatDType) (a_ptr dout_ptr db_ptr res_ptr : RegionName)
      (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
        stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m stride_dout_csize_n
        stride_db_batch stride_db_seqlen stride_db_head stride_db_k
        stride_res_batch stride_res_seqlen stride_res_head stride_res_k
        BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat) (HAS_RESIDUAL : Bool) =>
  triton {
    tl.ports inputs=[a_ptr, dout_ptr, res_ptr], outputs=[db_ptr]
    tl.region a_ptr = tl.float32, dout_ptr = tl.float32, res_ptr = DTYPE, db_ptr = OUT_DTYPE
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2)
    pid_c = pid_ch // $(ngroups)
    pid_h = pid_ch - pid_c * $(ngroups)
    num_pid_n = tl.cdiv($(K), $(BLOCK_SIZE_N))
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    a_ptr += pid_b * $(stride_a_batch) + pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
    dout_ptr += pid_b * $(stride_dout_batch) + pid_c * $(stride_dout_chunk) + pid_h * $(stride_dout_head)
    offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
    offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
    offs_cs = tl.arange(0, $(BLOCK_SIZE_CS))
    dout_ptrs = dout_ptr + (offs_m[:, None] * $(stride_dout_csize_n) + offs_cs[None, :] * $(stride_dout_csize_m))
    a_ptrs = a_ptr + (offs_cs[:, None] * $(stride_a_seqlen) + offs_n[None, :] * $(stride_ak))
    chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
    acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
    for cs in range($(0), tl.cdiv(chunk_size_limit, $(BLOCK_SIZE_CS)), $(1)) {
      dout = (tl.load(dout_ptrs, mask=(offs_m[:, None] < $(chunk_size)) &
        (offs_cs[None, :] < chunk_size_limit - cs * $(BLOCK_SIZE_CS)), other=0.0)).to(tl.float32)
      a = (tl.load(a_ptrs, mask=(offs_cs[:, None] < chunk_size_limit - cs * $(BLOCK_SIZE_CS)) &
        (offs_n[None, :] < $(K)), other=0.0)).to(tl.float32)
      acc += tl.dot(dout, a)
      dout_ptrs += $(BLOCK_SIZE_CS) * $(stride_dout_csize_m)
      a_ptrs += $(BLOCK_SIZE_CS) * $(stride_a_seqlen)
    }
    offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
    offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
    if HAS_RESIDUAL {
      res_ptr += pid_b * $(stride_res_batch) + pid_c * $(chunk_size) * $(stride_res_seqlen) + pid_h * $(stride_res_head)
      res_ptrs = res_ptr + (offs_m[:, None] * $(stride_res_seqlen) + offs_n[None, :] * $(stride_res_k))
      res = (tl.load(res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
        (offs_n[None, :] < $(K)))).to(tl.float32)
      acc += res
    }
    db = (acc).to(OUT_DTYPE)
    db_ptr += pid_b * $(stride_db_batch) + pid_c * $(chunk_size) * $(stride_db_seqlen) + pid_h * $(stride_db_head)
    db_ptrs = db_ptr + (offs_m[:, None] * $(stride_db_seqlen) + offs_n[None, :] * $(stride_db_k))
    tl.store(db_ptrs, db, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < $(K)))
  }

def quotedIndexCommuted :=
  fun (dtype out_dtype : FloatDType) (a_ptr dout_ptr db_ptr res_ptr : RegionName)
      (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
        stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m stride_dout_csize_n
        stride_db_batch stride_db_seqlen stride_db_head stride_db_k
        stride_res_batch stride_res_seqlen stride_res_head stride_res_k
        BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat) (HAS_RESIDUAL : Bool) =>
  triton {
    tl.ports inputs=[a_ptr, dout_ptr, res_ptr], outputs=[db_ptr]
    tl.region a_ptr = tl.float32, dout_ptr = tl.float32, res_ptr = DTYPE, db_ptr = OUT_DTYPE
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2)
    pid_c = pid_ch // $(ngroups)
    pid_h = pid_ch - $(ngroups) * pid_c
    num_pid_n = tl.cdiv($(K), $(BLOCK_SIZE_N))
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    a_ptr += pid_b * $(stride_a_batch) + pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
    dout_ptr += pid_b * $(stride_dout_batch) + pid_c * $(stride_dout_chunk) + pid_h * $(stride_dout_head)
    offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
    offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
    offs_cs = tl.arange(0, $(BLOCK_SIZE_CS))
    dout_ptrs = dout_ptr + (offs_m[:, None] * $(stride_dout_csize_n) + offs_cs[None, :] * $(stride_dout_csize_m))
    a_ptrs = a_ptr + (offs_cs[:, None] * $(stride_a_seqlen) + offs_n[None, :] * $(stride_ak))
    chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
    acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
    for cs in range($(0), tl.cdiv(chunk_size_limit, $(BLOCK_SIZE_CS)), $(1)) {
      dout = (tl.load(dout_ptrs, mask=(offs_m[:, None] < $(chunk_size)) &
        (offs_cs[None, :] < chunk_size_limit - cs * $(BLOCK_SIZE_CS)), other=0.0)).to(tl.float32)
      a = (tl.load(a_ptrs, mask=(offs_cs[:, None] < chunk_size_limit - cs * $(BLOCK_SIZE_CS)) &
        (offs_n[None, :] < $(K)), other=0.0)).to(tl.float32)
      acc += tl.dot(dout, a)
      dout_ptrs += $(BLOCK_SIZE_CS) * $(stride_dout_csize_m)
      a_ptrs += $(BLOCK_SIZE_CS) * $(stride_a_seqlen)
    }
    offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
    offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
    if HAS_RESIDUAL {
      res_ptr += pid_b * $(stride_res_batch) + pid_c * $(chunk_size) * $(stride_res_seqlen) + pid_h * $(stride_res_head)
      res_ptrs = res_ptr + (offs_m[:, None] * $(stride_res_seqlen) + offs_n[None, :] * $(stride_res_k))
      res = (tl.load(res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
        (offs_n[None, :] < $(K)))).to(tl.float32)
      acc += res
    }
    db = (acc).to(OUT_DTYPE)
    db_ptr += pid_b * $(stride_db_batch) + pid_c * $(chunk_size) * $(stride_db_seqlen) + pid_h * $(stride_db_head)
    db_ptrs = db_ptr + (offs_m[:, None] * $(stride_db_seqlen) + offs_n[None, :] * $(stride_db_k))
    tl.store(db_ptrs, db, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < $(K)))
  }

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
  simp only [fp32Source,bindMain,quotedOriginal,TypedSource.HostCall.source,ha,hd,TypedSource.specialize,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody]

#axiomsClean fp32Source_eq
end Kernels

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmChunkBwd
