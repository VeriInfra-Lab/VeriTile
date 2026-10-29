import bench.tritonbench_g.matmul_persistent_triton.Types
import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-!
Complete persistent source from TritonBench-G revision
603e28a (Python SHA256 4537e22452bcb9df7844521498b61232772d7ae44e9aa3220c38c2b341f2152c).
Both wrapper storage profiles retain the original grouped persistent schedule,
masked K tail, fused dot accumulator, conditional stores and zero resets.
The only candidate change commutes the exact output-row address product.
Integer antiquotations preserve the source's negative initial tile/ki values;
they do not replace source subtraction by truncated natural subtraction.
This helper is intentionally not a public corpus entry while proofs are built.
-/
namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.TypedSource
open Lean VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

syntax "persistentSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(persistentSource%[$dtype:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let outputRows : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M)))
      else `(tritonStmt| offs_cm = $(BLOCK_SIZE_M) * pid_m + tl.arange(0, $(BLOCK_SIZE_M)))
    `(fun (a_ptr b_ptr c_ptr : RegionName)
      (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M NUM_SMS : Nat) =>
      ComputeKernel.mk [a_ptr, b_ptr] [c_ptr] (triton {
        tl.region a_ptr = $dtype, b_ptr = $dtype, c_ptr = $dtype
        start_pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
        num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
        k_tiles = tl.cdiv($(K), $(BLOCK_SIZE_K))
        num_tiles = num_pid_m * num_pid_n
        tiles_per_SM = num_tiles // $(NUM_SMS)
        if start_pid < num_tiles % $(NUM_SMS) {
          tiles_per_SM += $(1)
        }
        tile_id = start_pid - $((NUM_SMS : Int))
        ki = $((-1 : Int))
        offs_k_for_mask = tl.arange(0, $(BLOCK_SIZE_K))
        num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
        pid_m = $((0 : Int))
        pid_n = $((0 : Int))
        offs_am = tl.arange(0, $(BLOCK_SIZE_M))
        offs_bn = tl.arange(0, $(BLOCK_SIZE_N))
        accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
        for _i in range($(0), k_tiles * tiles_per_SM, $(1)) {
          ki = tl.where(ki == k_tiles - $((1 : Int)), $((0 : Int)), ki + $((1 : Int)))
          if ki == $((0 : Int)) {
            tile_id += $((NUM_SMS : Int))
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * $(GROUP_SIZE_M)
            group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m
            start_m = pid_m * $(BLOCK_SIZE_M)
            start_n = pid_n * $(BLOCK_SIZE_N)
            offs_am = start_m + tl.arange(0, $(BLOCK_SIZE_M))
            offs_bn = start_n + tl.arange(0, $(BLOCK_SIZE_N))
            offs_am = tl.where(offs_am < $(M), offs_am, $((0 : Int)))
            offs_bn = tl.where(offs_bn < $(N), offs_bn, $((0 : Int)))
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, $(BLOCK_SIZE_M)), $(BLOCK_SIZE_M))
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, $(BLOCK_SIZE_N)), $(BLOCK_SIZE_N))
          }
          offs_k = ki * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K))
          a_ptrs = a_ptr + (offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak))
          b_ptrs = b_ptr + (offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
          a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < $(K) - ki * $(BLOCK_SIZE_K), other=0.0)
          b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < $(K) - ki * $(BLOCK_SIZE_K), other=0.0)
          accumulator = tl.dot(a, b, accumulator)
          if ki == k_tiles - $((1 : Int)) {
            $outputRows
            offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
            c_ptrs = c_ptr + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
            c = accumulator.to($dtype:tritonDType)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
          }
        }
      }).surfaceBody)


def kernel (storage : Storage) (candidate : Bool) : Source :=
  match storage, candidate with
  | .fp16, Bool.false => persistentSource%[tl.float16, original]
  | .fp16, Bool.true => persistentSource%[tl.float16, commuted]
  | .f8e4, Bool.false => persistentSource%[tl.float8e4nv, original]
  | .f8e4, Bool.true => persistentSource%[tl.float8e4nv, commuted]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.TypedSource
