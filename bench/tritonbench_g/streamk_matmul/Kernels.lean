import bench.tritonbench_g.streamk_matmul.Proofs.Support.SourceAdapter
import VeriTile.Triton.DSL.PastedControlFunctions
namespace VeriTile.Bench.Repairs.StreamK.TileChecked
open Reader VeriTile.Triton
set_option maxRecDepth 50000
set_option maxHeartbeats 8000000

def swizzle_tileSignature : CallSignature where
  pointers := []
  scalars := [("tile_id",.integer),("M",.integer),("N",.integer),("K",.integer)]
  constexpr := ["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"]
  readPorts := []
  writePorts := []

def linear_tileSignature : CallSignature where
  pointers := []
  scalars := [("tile_id",.integer),("M",.integer),("N",.integer),("K",.integer)]
  constexpr := ["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"]
  readPorts := []
  writePorts := []

def mac_loopSignature : CallSignature where
  pointers := [("A",.fp32),("B",.fp32),("C",.fp32),("partials",.fp32)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),("program",.integer),("stride_am",.integer),("stride_ak",.integer),("stride_bk",.integer),("stride_bn",.integer),("stride_cm",.integer),("stride_cn",.integer),("iters_per_tile",.integer),("start_iter",.integer),("end_iter",.integer)]
  constexpr := ["BLOCK_M", "BLOCK_N", "BLOCK_K", "ACC_TYPE", "GROUP_M"]
  readPorts := ["A", "B"]
  writePorts := ["C", "partials"]

def first_waveSignature : CallSignature where
  pointers := [("A",.fp32),("B",.fp32),("C",.fp32),("partials",.fp32)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),("stride_am",.integer),("stride_ak",.integer),("stride_bk",.integer),("stride_bn",.integer),("stride_cm",.integer),("stride_cn",.integer),("total_full_tiles_streamk",.integer),("total_partial_tiles_streamk",.integer),("iters_per_tile",.integer)]
  constexpr := ["BLOCK_M", "BLOCK_N", "BLOCK_K", "ACC_TYPE", "GROUP_M"]
  readPorts := ["A", "B"]
  writePorts := ["C", "partials"]

def full_tilesSignature : CallSignature where
  pointers := [("A",.fp32),("B",.fp32),("C",.fp32)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),("stride_am",.integer),("stride_ak",.integer),("stride_bk",.integer),("stride_bn",.integer),("stride_cm",.integer),("stride_cn",.integer),("total_tiles_streamk",.integer)]
  constexpr := ["BLOCK_M", "BLOCK_N", "BLOCK_K", "ACC_TYPE", "GROUP_M"]
  readPorts := ["A", "B"]
  writePorts := ["C"]

def reduce_streamkSignature : CallSignature where
  pointers := [("C",.fp32),("partials",.fp32)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),("stride_cm",.integer),("stride_cn",.integer),("total_programs_streamk",.integer),("total_full_tiles_streamk",.integer),("total_partial_tiles_streamk",.integer),("iters_per_tile",.integer)]
  constexpr := ["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"]
  readPorts := ["C", "partials"]
  writePorts := ["C", "partials"]

def swizzle_tileSource : VeriTile.Triton.DSL.PastedControlFunctions.Invocation :=
  with_control_function "swizzle_tile" swizzle_tileSignature constexpr_bools []
    dtype_constants [] jit_calls [] in triton {
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n
}

def linear_tileSource : VeriTile.Triton.DSL.PastedControlFunctions.Invocation :=
  with_control_function "linear_tile" linear_tileSignature constexpr_bools []
    dtype_constants [] jit_calls [] in triton {
    pid_m = tile_id // tl.cdiv(N, BLOCK_N)
    pid_n = tile_id % tl.cdiv(N, BLOCK_N)
    return pid_m, pid_n
}

def mac_loopSource : VeriTile.Triton.DSL.PastedControlFunctions.Invocation :=
  with_control_function "mac_loop" mac_loopSignature constexpr_bools []
    dtype_constants [ACC_TYPE := .fp32] jit_calls [swizzle_tile := "swizzle",linear_tile := "linear"] in triton {
    tile_id = start_iter // iters_per_tile
    if GROUP_M  > 0:
        pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K).to(tl.int64)
    step_a = tl.full((), BLOCK_K, tl.int64) * stride_ak
    step_b = tl.full((), BLOCK_K, tl.int64) * stride_bk
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak) + step_a * (start_iter % iters_per_tile)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn) + step_b * (start_iter % iters_per_tile)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for current_iter in range(start_iter, end_iter):
        kk = (current_iter % iters_per_tile) * BLOCK_K + rk
        a = tl.load(A, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B, mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        A += step_a
        B += step_b

    if end_iter % iters_per_tile == 0:  # unique terminal chunk initializes this tile
        C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)  # compute inside the if/else to avoid spilling!
        tl.store(C_, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
    else:
        # A program has at most one nonterminal chunk: its final chunk.
        slots = program * BLOCK_M * BLOCK_N + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        tl.store(partials + slots, acc)
}

def first_waveSource : VeriTile.Triton.DSL.PastedControlFunctions.Invocation :=
  with_control_function "first_wave" first_waveSignature constexpr_bools []
    dtype_constants [ACC_TYPE := .fp32] jit_calls [mac_loop := "mac"] in triton {
    pid = tl.program_id(0).to(tl.int64)
    start_iter = pid * total_full_tiles_streamk + tl.minimum(pid, total_partial_tiles_streamk)
    last_iter = (pid + 1) * total_full_tiles_streamk + tl.minimum(pid + 1, total_partial_tiles_streamk)

    while start_iter < last_iter:
        end_iter = tl.minimum(start_iter + (iters_per_tile - start_iter % iters_per_tile), last_iter)
        mac_loop(A, B, C,
                 M, N, K,
                 partials, pid,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 iters_per_tile,
                 start_iter, end_iter,
                 BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
                 GROUP_M,
                 )

        start_iter = end_iter
}

def full_tilesSource : VeriTile.Triton.DSL.PastedControlFunctions.Invocation :=
  with_control_function "full_tiles" full_tilesSignature constexpr_bools []
    dtype_constants [ACC_TYPE := .fp32] jit_calls [swizzle_tile := "swizzle",linear_tile := "linear"] in triton {
    tile_id = tl.program_id(0).to(tl.int64) + total_tiles_streamk
    if GROUP_M > 0:
        pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K).to(tl.int64)
    # Widen pointer increments before multiplying by the runtime strides.
    step_a = tl.full((), BLOCK_K, tl.int64) * stride_ak
    step_b = tl.full((), BLOCK_K, tl.int64) * stride_bk
    # pointers
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = (tl.full((), 0, tl.int64) + k) * BLOCK_K + rk
        a = tl.load(A, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B, mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        A += step_a
        B += step_b
    acc = acc.to(tl.float16)  # restore C.dtype.element_ty
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(C, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
}

def reduce_streamkSource : VeriTile.Triton.DSL.PastedControlFunctions.Invocation :=
  with_control_function "reduce_streamk" reduce_streamkSignature constexpr_bools []
    dtype_constants [] jit_calls [swizzle_tile := "swizzle",linear_tile := "linear"] in triton {
    tile_id = tl.program_id(0).to(tl.int64)
    if GROUP_M > 0:
        pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    for program in range(0, total_programs_streamk):
        pid = tl.full((), 0, tl.int64) + program
        begin = pid * total_full_tiles_streamk + tl.minimum(pid, total_partial_tiles_streamk)
        end = (pid + 1) * total_full_tiles_streamk + tl.minimum(pid + 1, total_partial_tiles_streamk)
        if (begin < end) & (end % iters_per_tile != 0) & (end // iters_per_tile == tile_id):
            slots = pid * BLOCK_M * BLOCK_N + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
            contribution = tl.load(partials + slots)
            tl.atomic_add(C + offsets, contribution, mask=mask)
}

def area_swizzle_tileSource : VeriTile.Triton.DSL.PastedControlFunctions.Invocation :=
  with_control_function "swizzle_tile_area" swizzle_tileSignature constexpr_bools []
    dtype_constants [] jit_calls [] in triton {
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n
}

def area_linear_tileSource : VeriTile.Triton.DSL.PastedControlFunctions.Invocation :=
  with_control_function "linear_tile_area" linear_tileSignature constexpr_bools []
    dtype_constants [] jit_calls [] in triton {
    pid_m = tile_id // tl.cdiv(N, BLOCK_N)
    pid_n = tile_id % tl.cdiv(N, BLOCK_N)
    return pid_m, pid_n
}

def area_mac_loopSource : VeriTile.Triton.DSL.PastedControlFunctions.Invocation :=
  with_control_function "mac_loop_area" mac_loopSignature constexpr_bools []
    dtype_constants [ACC_TYPE := .fp32] jit_calls [swizzle_tile_area := "swizzle",linear_tile_area := "linear"] in triton {
    tile_id = start_iter // iters_per_tile
    if GROUP_M  > 0:
        pid_m, pid_n = swizzle_tile_area(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid_m, pid_n = linear_tile_area(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K).to(tl.int64)
    step_a = tl.full((), BLOCK_K, tl.int64) * stride_ak
    step_b = tl.full((), BLOCK_K, tl.int64) * stride_bk
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak) + step_a * (start_iter % iters_per_tile)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn) + step_b * (start_iter % iters_per_tile)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for current_iter in range(start_iter, end_iter):
        kk = (current_iter % iters_per_tile) * BLOCK_K + rk
        a = tl.load(A, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B, mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        A += step_a
        B += step_b

    if end_iter % iters_per_tile == 0:  # unique terminal chunk initializes this tile
        C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)  # compute inside the if/else to avoid spilling!
        tl.store(C_, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
    else:
        # A program has at most one nonterminal chunk: its final chunk.
        slots = program * (BLOCK_M * BLOCK_N) + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        tl.store(partials + slots, acc)
}

def area_first_waveSource : VeriTile.Triton.DSL.PastedControlFunctions.Invocation :=
  with_control_function "first_wave_area" first_waveSignature constexpr_bools []
    dtype_constants [ACC_TYPE := .fp32] jit_calls [mac_loop_area := "mac"] in triton {
    pid = tl.program_id(0).to(tl.int64)
    start_iter = pid * total_full_tiles_streamk + tl.minimum(pid, total_partial_tiles_streamk)
    last_iter = (pid + 1) * total_full_tiles_streamk + tl.minimum(pid + 1, total_partial_tiles_streamk)

    while start_iter < last_iter:
        end_iter = tl.minimum(start_iter + (iters_per_tile - start_iter % iters_per_tile), last_iter)
        mac_loop_area(A, B, C,
                 M, N, K,
                 partials, pid,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 iters_per_tile,
                 start_iter, end_iter,
                 BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
                 GROUP_M,
                 )

        start_iter = end_iter
}

def area_full_tilesSource : VeriTile.Triton.DSL.PastedControlFunctions.Invocation :=
  with_control_function "full_tiles_area" full_tilesSignature constexpr_bools []
    dtype_constants [ACC_TYPE := .fp32] jit_calls [swizzle_tile_area := "swizzle",linear_tile_area := "linear"] in triton {
    tile_id = tl.program_id(0).to(tl.int64) + total_tiles_streamk
    if GROUP_M > 0:
        pid_m, pid_n = swizzle_tile_area(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid_m, pid_n = linear_tile_area(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K).to(tl.int64)
    # Widen pointer increments before multiplying by the runtime strides.
    step_a = tl.full((), BLOCK_K, tl.int64) * stride_ak
    step_b = tl.full((), BLOCK_K, tl.int64) * stride_bk
    # pointers
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = (tl.full((), 0, tl.int64) + k) * BLOCK_K + rk
        a = tl.load(A, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B, mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        A += step_a
        B += step_b
    acc = acc.to(tl.float16)  # restore C.dtype.element_ty
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(C, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
}

def area_reduce_streamkSource : VeriTile.Triton.DSL.PastedControlFunctions.Invocation :=
  with_control_function "reduce_streamk_area" reduce_streamkSignature constexpr_bools []
    dtype_constants [] jit_calls [swizzle_tile_area := "swizzle",linear_tile_area := "linear"] in triton {
    tile_id = tl.program_id(0).to(tl.int64)
    if GROUP_M > 0:
        pid_m, pid_n = swizzle_tile_area(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid_m, pid_n = linear_tile_area(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    for program in range(0, total_programs_streamk):
        pid = tl.full((), 0, tl.int64) + program
        begin = pid * total_full_tiles_streamk + tl.minimum(pid, total_partial_tiles_streamk)
        end = (pid + 1) * total_full_tiles_streamk + tl.minimum(pid + 1, total_partial_tiles_streamk)
        if (begin < end) & (end % iters_per_tile != 0) & (end // iters_per_tile == tile_id):
            slots = pid * (BLOCK_M * BLOCK_N) + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
            contribution = tl.load(partials + slots)
            tl.atomic_add(C + offsets, contribution, mask=mask)
}

abbrev swizzle_tileDecoded : {body : List Statement // SourceAdapter.decode swizzle_tileSource=.ok body} := by
  refine ⟨?_,?_⟩
  case refine_2 => rfl
def swizzle_tile : List Statement := swizzle_tileDecoded.val

abbrev linear_tileDecoded : {body : List Statement // SourceAdapter.decode linear_tileSource=.ok body} := by
  refine ⟨?_,?_⟩
  case refine_2 => rfl
def linear_tile : List Statement := linear_tileDecoded.val

abbrev mac_loopDecoded : {body : List Statement // SourceAdapter.decode mac_loopSource=.ok body} := by
  refine ⟨?_,?_⟩
  case refine_2 => rfl
def mac_loop : List Statement := mac_loopDecoded.val

abbrev first_waveDecoded : {body : List Statement // SourceAdapter.decode first_waveSource=.ok body} := by
  refine ⟨?_,?_⟩
  case refine_2 => rfl
def first_wave : List Statement := first_waveDecoded.val

abbrev full_tilesDecoded : {body : List Statement // SourceAdapter.decode full_tilesSource=.ok body} := by
  refine ⟨?_,?_⟩
  case refine_2 => rfl
def full_tiles : List Statement := full_tilesDecoded.val

abbrev reduce_streamkDecoded : {body : List Statement // SourceAdapter.decode reduce_streamkSource=.ok body} := by
  refine ⟨?_,?_⟩
  case refine_2 => rfl
def reduce_streamk : List Statement := reduce_streamkDecoded.val

abbrev area_swizzle_tileDecoded : {body : List Statement // SourceAdapter.decode area_swizzle_tileSource=.ok body} := by
  refine ⟨?_,?_⟩
  case refine_2 => rfl
def area_swizzle_tile : List Statement := area_swizzle_tileDecoded.val

abbrev area_linear_tileDecoded : {body : List Statement // SourceAdapter.decode area_linear_tileSource=.ok body} := by
  refine ⟨?_,?_⟩
  case refine_2 => rfl
def area_linear_tile : List Statement := area_linear_tileDecoded.val

abbrev area_mac_loopDecoded : {body : List Statement // SourceAdapter.decode area_mac_loopSource=.ok body} := by
  refine ⟨?_,?_⟩
  case refine_2 => rfl
def area_mac_loop : List Statement := area_mac_loopDecoded.val

abbrev area_first_waveDecoded : {body : List Statement // SourceAdapter.decode area_first_waveSource=.ok body} := by
  refine ⟨?_,?_⟩
  case refine_2 => rfl
def area_first_wave : List Statement := area_first_waveDecoded.val

abbrev area_full_tilesDecoded : {body : List Statement // SourceAdapter.decode area_full_tilesSource=.ok body} := by
  refine ⟨?_,?_⟩
  case refine_2 => rfl
def area_full_tiles : List Statement := area_full_tilesDecoded.val

abbrev area_reduce_streamkDecoded : {body : List Statement // SourceAdapter.decode area_reduce_streamkSource=.ok body} := by
  refine ⟨?_,?_⟩
  case refine_2 => rfl
def area_reduce_streamk : List Statement := area_reduce_streamkDecoded.val

abbrev kernelsDecoded : {body : Sources // SourceAdapter.decodeSources swizzle_tileSource linear_tileSource mac_loopSource first_waveSource full_tilesSource reduce_streamkSource=.ok body} := by
  refine ⟨?_,?_⟩
  case refine_2 => rfl
def kernels : Sources := kernelsDecoded.val

abbrev areaKernelsDecoded : {body : Sources // SourceAdapter.decodeSources area_swizzle_tileSource area_linear_tileSource area_mac_loopSource area_first_waveSource area_full_tilesSource area_reduce_streamkSource=.ok body} := by
  refine ⟨?_,?_⟩
  case refine_2 => rfl
def areaKernels : Sources := areaKernelsDecoded.val

end VeriTile.Bench.Repairs.StreamK.TileChecked
