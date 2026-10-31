import bench.tritonbench_g.int_scaled_matmul.Sources.Parameters

/-! Direct source for the selected signed8 / modular-add repair. The explicit
casts retain byte storage and the zero-seeded dot followed by add.u32. The
related candidates remove a copied accumulator and repeated row/column indices.
No measured speedup is claimed. -/
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Kernels
open VeriTile.Triton
open Revision001
noncomputable section
def unscaledSource
    (a_ptr b_ptr : Region .int) (c_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) :
    ComputeKernel := triton {
  tl.ports inputs=[Region.cast a_ptr, Region.cast b_ptr], outputs=[c_ptr]
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_N))
  num_pid_in_group = $(GROUP_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_M)
  GROUP_M = min(num_pid_m - first_pid_m, $(GROUP_M))
  pid_m = first_pid_m + (pid % GROUP_M)
  pid_n = (pid % num_pid_in_group) // GROUP_M
  a_block_ptr = tl.make_block_ptr(base=$((a_ptr : Region .int)), shape=($(M), $(K)),
    strides=($(stride_am), $(stride_ak)), offsets=(pid_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_K)), order=(1, 0))
  b_block_ptr = tl.make_block_ptr(base=$((b_ptr : Region .int)), shape=($(K), $(N)),
    strides=($(stride_bk), $(stride_bn)), offsets=(0, pid_n * $(BLOCK_N)),
    block_shape=($(BLOCK_K), $(BLOCK_N)), order=(1, 0))
  accumulator = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.int32)
  for k in range(0, $(K), $(BLOCK_K)) {
    a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.int8)
    b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.int8)
    accumulator = (accumulator + tl.dot(a, b)).to(tl.int32)
    a_block_ptr = tl.advance(a_block_ptr, [$(0), $(BLOCK_K)])
    b_block_ptr = tl.advance(b_block_ptr, [$(BLOCK_K), $(0)])
  }
  c = accumulator
  c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=($(M), $(N)),
    strides=($(stride_cm), $(stride_cn)),
    offsets=(pid_m * $(BLOCK_M), pid_n * $(BLOCK_N)),
    block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
  tl.store(c_block_ptr, c, boundary_check=(0, 1))
}

def unscaledCopyFreeSource
    (a_ptr b_ptr : Region .int) (c_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) :
    ComputeKernel := triton {
  tl.ports inputs=[Region.cast a_ptr, Region.cast b_ptr], outputs=[c_ptr]
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_N))
  num_pid_in_group = $(GROUP_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_M)
  GROUP_M = min(num_pid_m - first_pid_m, $(GROUP_M))
  pid_m = first_pid_m + (pid % GROUP_M)
  pid_n = (pid % num_pid_in_group) // GROUP_M
  a_block_ptr = tl.make_block_ptr(base=$((a_ptr : Region .int)), shape=($(M), $(K)),
    strides=($(stride_am), $(stride_ak)), offsets=(pid_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_K)), order=(1, 0))
  b_block_ptr = tl.make_block_ptr(base=$((b_ptr : Region .int)), shape=($(K), $(N)),
    strides=($(stride_bk), $(stride_bn)), offsets=(0, pid_n * $(BLOCK_N)),
    block_shape=($(BLOCK_K), $(BLOCK_N)), order=(1, 0))
  accumulator = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.int32)
  for k in range(0, $(K), $(BLOCK_K)) {
    a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.int8)
    b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.int8)
    accumulator = (accumulator + tl.dot(a, b)).to(tl.int32)
    a_block_ptr = tl.advance(a_block_ptr, [$(0), $(BLOCK_K)])
    b_block_ptr = tl.advance(b_block_ptr, [$(BLOCK_K), $(0)])
  }
  c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=($(M), $(N)),
    strides=($(stride_cm), $(stride_cn)),
    offsets=(pid_m * $(BLOCK_M), pid_n * $(BLOCK_N)),
    block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
  tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))
}

def scaledSource
    (a_ptr b_ptr : Region .int) (c_ptr s1_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_s1m stride_s1n : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) (EVEN_K : Bool)
    (numKBlocks : Nat) :
    ComputeKernel := triton {
  tl.ports inputs=[Region.cast a_ptr, Region.cast b_ptr, s1_ptr], outputs=[c_ptr]
  tl.region s1_ptr = tl.float32, c_ptr = tl.int32
  pid = tl.program_id(0)
  grid_m = ($((M : Nat)) + $(BLOCK_M) - $(1)) // $(BLOCK_M)
  grid_n = ($((N : Nat)) + $(BLOCK_N) - $(1)) // $(BLOCK_N)
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // (group_size)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  ram = rm % $(M)
  rbn = rn % $(N)
  rk = tl.arange(0, $(BLOCK_K))
  A = $((a_ptr : Region .int)) + (ram[:, None] * $(stride_am) + rk[None, :] * $(stride_ak))
  B = $((b_ptr : Region .int)) + (rk[:, None] * $(stride_bk) + rbn[None, :] * $(stride_bn))
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.int32)
  for j in range(0, $(numKBlocks), $(1)) {
    k = $(K) - j * $(BLOCK_K)
    if EVEN_K {
      a = tl.load(A).to(tl.int8)
      b = tl.load(B).to(tl.int8)
    } else {
      a = tl.load(A, mask=rk[None, :] < k, other=0.0).to(tl.int8)
      b = tl.load(B, mask=rk[:, None] < k, other=0.0).to(tl.int8)
    }
    acc = (acc + tl.dot(a, b)).to(tl.int32)
    A += $(BLOCK_K) * $(stride_ak)
    B += $(BLOCK_K) * $(stride_bk)
  }
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  idx_m = rm[:, None]
  idx_n = rn[None, :]
  mask = (idx_m < $(M)) & (idx_n < $(N))
  xindex = idx_n * $(stride_cn) + idx_m * $(stride_cm)
  tmp0 = tl.load(s1_ptr + (tl.broadcast_to(idx_m * $(stride_s1m), [$(BLOCK_M), $(BLOCK_N)])), mask, eviction_policy="evict_last").to(tl.float32)
  tl.store(c_ptr + (tl.broadcast_to(xindex, [$(BLOCK_M), $(BLOCK_N)])), (acc * tmp0).to(tl.int32), mask)
}

def scaledSharedIndicesSource
    (a_ptr b_ptr : Region .int) (c_ptr s1_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_s1m stride_s1n : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) (EVEN_K : Bool)
    (numKBlocks : Nat) :
    ComputeKernel := triton {
  tl.ports inputs=[Region.cast a_ptr, Region.cast b_ptr, s1_ptr], outputs=[c_ptr]
  tl.region s1_ptr = tl.float32, c_ptr = tl.int32
  pid = tl.program_id(0)
  grid_m = ($((M : Nat)) + $(BLOCK_M) - $(1)) // $(BLOCK_M)
  grid_n = ($((N : Nat)) + $(BLOCK_N) - $(1)) // $(BLOCK_N)
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // (group_size)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  ram = rm % $(M)
  rbn = rn % $(N)
  rk = tl.arange(0, $(BLOCK_K))
  A = $((a_ptr : Region .int)) + (ram[:, None] * $(stride_am) + rk[None, :] * $(stride_ak))
  B = $((b_ptr : Region .int)) + (rk[:, None] * $(stride_bk) + rbn[None, :] * $(stride_bn))
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.int32)
  for j in range(0, $(numKBlocks), $(1)) {
    k = $(K) - j * $(BLOCK_K)
    if EVEN_K {
      a = tl.load(A).to(tl.int8)
      b = tl.load(B).to(tl.int8)
    } else {
      a = tl.load(A, mask=rk[None, :] < k, other=0.0).to(tl.int8)
      b = tl.load(B, mask=rk[:, None] < k, other=0.0).to(tl.int8)
    }
    acc = (acc + tl.dot(a, b)).to(tl.int32)
    A += $(BLOCK_K) * $(stride_ak)
    B += $(BLOCK_K) * $(stride_bk)
  }
  idx_m = rm[:, None]
  idx_n = rn[None, :]
  mask = (idx_m < $(M)) & (idx_n < $(N))
  xindex = idx_n * $(stride_cn) + idx_m * $(stride_cm)
  tmp0 = tl.load(s1_ptr + (tl.broadcast_to(idx_m * $(stride_s1m), [$(BLOCK_M), $(BLOCK_N)])), mask, eviction_policy="evict_last").to(tl.float32)
  tl.store(c_ptr + (tl.broadcast_to(xindex, [$(BLOCK_M), $(BLOCK_N)])), (acc * tmp0).to(tl.int32), mask)
}

def unscaled (p : BlockWord.Parameters) : ComputeKernel :=
  unscaledSource
  (Region.cast p.input) (Region.cast p.weight) p.output p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns p.blockInner p.groupRows

def unscaledCopyFree (p : BlockWord.Parameters) : ComputeKernel :=
  unscaledCopyFreeSource
  (Region.cast p.input) (Region.cast p.weight) p.output p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns p.blockInner p.groupRows

def scaled (p : ScaledWord.Parameters) : ComputeKernel :=
  scaledSource
  (Region.cast p.input) (Region.cast p.weight) p.output p.scale p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.scaleRowStride p.scaleColumnStride
  p.blockRows p.blockColumns p.blockInner p.groupRows p.even p.steps

def scaledSharedIndices (p : ScaledWord.Parameters) : ComputeKernel :=
  scaledSharedIndicesSource
  (Region.cast p.input) (Region.cast p.weight) p.output p.scale p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.scaleRowStride p.scaleColumnStride
  p.blockRows p.blockColumns p.blockInner p.groupRows p.even p.steps

end
end VeriTile.Bench.Repairs.IntScaledMatmul.Kernels
