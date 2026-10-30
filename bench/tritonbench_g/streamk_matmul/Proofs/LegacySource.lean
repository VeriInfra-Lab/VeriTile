import bench.tritonbench_g.streamk_matmul.Proofs.LegacySyntax

/-! Frozen whole-program preservation fixtures. The canonical sources live
in Kernels.lean and use the shared control-flow frontend.

Readable source for the independent static-domain candidate. The six
kernel bodies are unchanged from the frozen parent. ACC_TYPE is FP32 by
the public contract; method casts use tl.cast and shape tuples use brackets.
The local reader retains both mapping helpers and all helper arguments. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.SourceLegacy
open Reader Syntax

def swizzle_tile : List Statement := streamk_triton {
  grid_m = tl.cdiv(M, BLOCK_M)
  grid_n = tl.cdiv(N, BLOCK_N)
  width = (GROUP_M * grid_n)
  group_id = (tile_id // width)
  group_size = tl.minimum((grid_m - (group_id * GROUP_M)), GROUP_M)
  pid_m = ((group_id * GROUP_M) + (tile_id % group_size))
  pid_n = ((tile_id % width) // group_size)
  return pid_m, pid_n
}

def linear_tile : List Statement := streamk_triton {
  pid_m = (tile_id // tl.cdiv(N, BLOCK_N))
  pid_n = (tile_id % tl.cdiv(N, BLOCK_N))
  return pid_m, pid_n
}

def mac_loop : List Statement := streamk_triton {
  tile_id = (start_iter // iters_per_tile)
  if (GROUP_M > 0) {
    pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
  } else {
    pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
  }
  rm = ((pid_m * BLOCK_M) + tl.arange(0, BLOCK_M))
  rn = ((pid_n * BLOCK_N) + tl.arange(0, BLOCK_N))
  rk = tl.cast(tl.arange(0, BLOCK_K), tl.int64)
  step_a = (tl.full((), BLOCK_K, tl.int64) * stride_ak)
  step_b = (tl.full((), BLOCK_K, tl.int64) * stride_bk)
  A = ((A + ((rm[:, None] * stride_am) + (rk[None, :] * stride_ak))) + (step_a * (start_iter % iters_per_tile)))
  B = ((B + ((rk[:, None] * stride_bk) + (rn[None, :] * stride_bn))) + (step_b * (start_iter % iters_per_tile)))
  acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
  for current_iter in range(start_iter, end_iter) {
    kk = (((current_iter % iters_per_tile) * BLOCK_K) + rk)
    a = tl.load(A, mask=((rm[:, None] < M) & (kk[None, :] < K)), other=0.0)
    b = tl.load(B, mask=((kk[:, None] < K) & (rn[None, :] < N)), other=0.0)
    acc += tl.dot(a, b)
    A += step_a
    B += step_b
  }
  if ((end_iter % iters_per_tile) == 0) {
    C_ = (C + ((rm[:, None] * stride_cm) + (rn[None, :] * stride_cn)))
    tl.store(C_, acc, mask=((rm[:, None] < M) & (rn[None, :] < N)))
  } else {
    slots = ((((program * BLOCK_M) * BLOCK_N) + (tl.arange(0, BLOCK_M)[:, None] * BLOCK_N)) + tl.arange(0, BLOCK_N)[None, :])
    tl.store((partials + slots), acc)
  }
}

def first_wave : List Statement := streamk_triton {
  pid = tl.cast(tl.program_id(0), tl.int64)
  start_iter = ((pid * total_full_tiles_streamk) + tl.minimum(pid, total_partial_tiles_streamk))
  last_iter = (((pid + 1) * total_full_tiles_streamk) + tl.minimum((pid + 1), total_partial_tiles_streamk))
  while (start_iter < last_iter) {
    end_iter = tl.minimum((start_iter + (iters_per_tile - (start_iter % iters_per_tile))), last_iter)
    mac_loop(A, B, C, M, N, K, partials, pid, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, start_iter, end_iter, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M)
    start_iter = end_iter
  }
}

def full_tiles : List Statement := streamk_triton {
  tile_id = (tl.cast(tl.program_id(0), tl.int64) + total_tiles_streamk)
  if (GROUP_M > 0) {
    pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
  } else {
    pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
  }
  rm = ((pid_m * BLOCK_M) + tl.arange(0, BLOCK_M))
  rn = ((pid_n * BLOCK_N) + tl.arange(0, BLOCK_N))
  rk = tl.cast(tl.arange(0, BLOCK_K), tl.int64)
  step_a = (tl.full((), BLOCK_K, tl.int64) * stride_ak)
  step_b = (tl.full((), BLOCK_K, tl.int64) * stride_bk)
  A = (A + ((rm[:, None] * stride_am) + (rk[None, :] * stride_ak)))
  B = (B + ((rk[:, None] * stride_bk) + (rn[None, :] * stride_bn)))
  acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
  for k in range(0, tl.cdiv(K, BLOCK_K)) {
    kk = (((tl.full((), 0, tl.int64) + k) * BLOCK_K) + rk)
    a = tl.load(A, mask=((rm[:, None] < M) & (kk[None, :] < K)), other=0.0)
    b = tl.load(B, mask=((kk[:, None] < K) & (rn[None, :] < N)), other=0.0)
    acc += tl.dot(a, b)
    A += step_a
    B += step_b
  }
  acc = tl.cast(acc, tl.float16)
  rm = ((pid_m * BLOCK_M) + tl.arange(0, BLOCK_M))
  rn = ((pid_n * BLOCK_N) + tl.arange(0, BLOCK_N))
  C = (C + ((rm[:, None] * stride_cm) + (rn[None, :] * stride_cn)))
  tl.store(C, acc, mask=((rm[:, None] < M) & (rn[None, :] < N)))
}

def reduce_streamk : List Statement := streamk_triton {
  tile_id = tl.cast(tl.program_id(0), tl.int64)
  if (GROUP_M > 0) {
    pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
  } else {
    pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
  }
  rm = ((pid_m * BLOCK_M) + tl.arange(0, BLOCK_M))
  rn = ((pid_n * BLOCK_N) + tl.arange(0, BLOCK_N))
  offsets = ((rm[:, None] * stride_cm) + (rn[None, :] * stride_cn))
  mask = ((rm[:, None] < M) & (rn[None, :] < N))
  for program in range(0, total_programs_streamk) {
    pid = (tl.full((), 0, tl.int64) + program)
    begin = ((pid * total_full_tiles_streamk) + tl.minimum(pid, total_partial_tiles_streamk))
    end = (((pid + 1) * total_full_tiles_streamk) + tl.minimum((pid + 1), total_partial_tiles_streamk))
    if (((begin < end) & ((end % iters_per_tile) != 0)) & ((end // iters_per_tile) == tile_id)) {
      slots = ((((pid * BLOCK_M) * BLOCK_N) + (tl.arange(0, BLOCK_M)[:, None] * BLOCK_N)) + tl.arange(0, BLOCK_N)[None, :])
      contribution = tl.load((partials + slots))
      tl.atomic_add((C + offsets), contribution, mask=mask)
    }
  }
}

/-! Actual candidates fold the constexpr tile area before the live program
index multiply. Four complete unchanged helper/stage quotations are shared. -/
def area_mac_loop : List Statement := streamk_triton {
  tile_id = (start_iter // iters_per_tile)
  if (GROUP_M > 0) {
    pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
  } else {
    pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
  }
  rm = ((pid_m * BLOCK_M) + tl.arange(0, BLOCK_M))
  rn = ((pid_n * BLOCK_N) + tl.arange(0, BLOCK_N))
  rk = tl.cast(tl.arange(0, BLOCK_K), tl.int64)
  step_a = (tl.full((), BLOCK_K, tl.int64) * stride_ak)
  step_b = (tl.full((), BLOCK_K, tl.int64) * stride_bk)
  A = ((A + ((rm[:, None] * stride_am) + (rk[None, :] * stride_ak))) + (step_a * (start_iter % iters_per_tile)))
  B = ((B + ((rk[:, None] * stride_bk) + (rn[None, :] * stride_bn))) + (step_b * (start_iter % iters_per_tile)))
  acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
  for current_iter in range(start_iter, end_iter) {
    kk = (((current_iter % iters_per_tile) * BLOCK_K) + rk)
    a = tl.load(A, mask=((rm[:, None] < M) & (kk[None, :] < K)), other=0.0)
    b = tl.load(B, mask=((kk[:, None] < K) & (rn[None, :] < N)), other=0.0)
    acc += tl.dot(a, b)
    A += step_a
    B += step_b
  }
  if ((end_iter % iters_per_tile) == 0) {
    C_ = (C + ((rm[:, None] * stride_cm) + (rn[None, :] * stride_cn)))
    tl.store(C_, acc, mask=((rm[:, None] < M) & (rn[None, :] < N)))
  } else {
    slots = (((program * (BLOCK_M * BLOCK_N)) + (tl.arange(0, BLOCK_M)[:, None] * BLOCK_N)) + tl.arange(0, BLOCK_N)[None, :])
    tl.store((partials + slots), acc)
  }
}

def area_reduce_streamk : List Statement := streamk_triton {
  tile_id = tl.cast(tl.program_id(0), tl.int64)
  if (GROUP_M > 0) {
    pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
  } else {
    pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
  }
  rm = ((pid_m * BLOCK_M) + tl.arange(0, BLOCK_M))
  rn = ((pid_n * BLOCK_N) + tl.arange(0, BLOCK_N))
  offsets = ((rm[:, None] * stride_cm) + (rn[None, :] * stride_cn))
  mask = ((rm[:, None] < M) & (rn[None, :] < N))
  for program in range(0, total_programs_streamk) {
    pid = (tl.full((), 0, tl.int64) + program)
    begin = ((pid * total_full_tiles_streamk) + tl.minimum(pid, total_partial_tiles_streamk))
    end = (((pid + 1) * total_full_tiles_streamk) + tl.minimum((pid + 1), total_partial_tiles_streamk))
    if (((begin < end) & ((end % iters_per_tile) != 0)) & ((end // iters_per_tile) == tile_id)) {
      slots = (((pid * (BLOCK_M * BLOCK_N)) + (tl.arange(0, BLOCK_M)[:, None] * BLOCK_N)) + tl.arange(0, BLOCK_N)[None, :])
      contribution = tl.load((partials + slots))
      tl.atomic_add((C + offsets), contribution, mask=mask)
    }
  }
}

def kernels : Sources := ⟨swizzle_tile,linear_tile,mac_loop,first_wave,full_tiles,reduce_streamk⟩
def areaKernels : Sources :=
  ⟨swizzle_tile,linear_tile,area_mac_loop,first_wave,full_tiles,area_reduce_streamk⟩
end VeriTile.Bench.Repairs.StreamK.TileChecked.SourceLegacy
