import bench.tritonbench_g.matmul_persistent_triton.Proofs.Source

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton.TypedSource
open Lean VeriTile.Triton
abbrev Storage := VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.TypedSource.Storage
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

/-- Both production profiles retain the fused dot and actual output cast.
The repaired persistent loop has an outer assigned-tile loop and inner K loop.
Masked coordinates are clamped before pointer formation, without false hints.
Boolean-to-integer additions are displayed as exact conditional increments. -/
syntax "persistentRepaired%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(persistentRepaired%[$dtype:tritonDType]) =>
    `(fun (a_ptr b_ptr c_ptr : RegionName)
      (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M NUM_SMS : Nat) =>
      ComputeKernel.mk [a_ptr,b_ptr] [c_ptr] (triton {
        tl.region a_ptr = $dtype, b_ptr = $dtype, c_ptr = $dtype
        start_pid = tl.program_id(axis=0).to(tl.int64)
        num_pid_m = $(M) // $(BLOCK_SIZE_M)
        if $(M) % $(BLOCK_SIZE_M) != $(0) {
          num_pid_m += $(1)
        }
        num_pid_n = $(N) // $(BLOCK_SIZE_N)
        if $(N) % $(BLOCK_SIZE_N) != $(0) {
          num_pid_n += $(1)
        }
        k_tiles = $(K) // $(BLOCK_SIZE_K)
        if $(K) % $(BLOCK_SIZE_K) != $(0) {
          k_tiles += $(1)
        }
        num_tiles = num_pid_m * num_pid_n
        tiles_per_SM = num_tiles // $(NUM_SMS)
        if start_pid < num_tiles % $(NUM_SMS) {
          tiles_per_SM += $(1)
        }
        active_group_m = min($(GROUP_SIZE_M),num_pid_m)
        num_pid_in_group = active_group_m * num_pid_n
        lanes_m = tl.arange(0,$(BLOCK_SIZE_M)).to(tl.int64)
        lanes_n = tl.arange(0,$(BLOCK_SIZE_N)).to(tl.int64)
        offs_k_for_mask = tl.arange(0,$(BLOCK_SIZE_K)).to(tl.int64)
        for tile_ordinal in range($(0),tiles_per_SM,$(1)) {
          tile_id = start_pid + tile_ordinal.to(tl.int64) * $(NUM_SMS)
          group_id = tile_id // num_pid_in_group
          first_pid_m = group_id * active_group_m
          group_size_m = min(num_pid_m - first_pid_m,active_group_m)
          pid_m = first_pid_m + (tile_id % group_size_m)
          pid_n = (tile_id % num_pid_in_group) // group_size_m
          start_m = pid_m * $(BLOCK_SIZE_M)
          start_n = pid_n * $(BLOCK_SIZE_N)
          offs_cm = start_m + lanes_m
          offs_cn = start_n + lanes_n
          offs_am = tl.where(offs_cm < $(M),offs_cm,$((0 : Int)))
          offs_bn = tl.where(offs_cn < $(N),offs_cn,$((0 : Int)))
          accumulator = tl.zeros([$(BLOCK_SIZE_M),$(BLOCK_SIZE_N)],dtype=tl.float32)
          for ki in range($(0),k_tiles,$(1)) {
            raw_k = ki.to(tl.int64) * $(BLOCK_SIZE_K) + offs_k_for_mask
            offs_k = tl.where(raw_k < $(K),raw_k,$((0 : Int)))
            a_ptrs = a_ptr + (offs_am[:,None] * ($(stride_am)).to(tl.int64) + offs_k[None,:] * ($(stride_ak)).to(tl.int64))
            b_ptrs = b_ptr + (offs_k[:,None] * ($(stride_bk)).to(tl.int64) + offs_bn[None,:] * ($(stride_bn)).to(tl.int64))
            a = tl.load(a_ptrs,mask=offs_k_for_mask[None,:] < $(K) - ki.to(tl.int64) * $(BLOCK_SIZE_K),other=0.0)
            b = tl.load(b_ptrs,mask=offs_k_for_mask[:,None] < $(K) - ki.to(tl.int64) * $(BLOCK_SIZE_K),other=0.0)
            accumulator = tl.dot(a,b,accumulator)
          }
          c_ptrs = c_ptr + ($(stride_cm)).to(tl.int64) * offs_am[:,None] + ($(stride_cn)).to(tl.int64) * offs_bn[None,:]
          c_mask = (offs_cm[:,None] < $(M)) & (offs_cn[None,:] < $(N))
          c = accumulator.to($dtype:tritonDType)
          tl.store(c_ptrs,c,mask=c_mask)
        }
      }).surfaceBody)

def kernel : Storage → VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.TypedSource.Source
  | .fp16 => persistentRepaired%[tl.float16]
  | .f8e4 => persistentRepaired%[tl.float8e4nv]

end VeriTile.Bench.Repairs.MatmulPersistentTriton.TypedSource
