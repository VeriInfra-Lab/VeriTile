import bench.tritonbench_g.kv_cache_copy.Proofs.Sources.Parameters
import VeriTile.Triton.DSL.StaticRanges

/-! Complete original and existing offset-commuted Python bodies. Caller typing
and runtime bindings precede the quotations; no source directive is inserted. -/
namespace VeriTile.Bench.Repairs.KvCacheCopy.Source
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("K",.fp32), ("V",.fp32), ("KCache",.fp32), ("VCache",.fp32), ("BLOCK_TABLES",.i32), ("context_lengths",.i32)]
  scalars := [("stride_kt",.integer), ("stride_kh",.integer), ("stride_kd",.integer), ("stride_vt",.integer), ("stride_vh",.integer), ("stride_vd",.integer), ("stride_kcb",.integer), ("stride_kch",.integer), ("stride_kcsplit_x",.integer), ("stride_kcs",.integer), ("stride_kcd",.integer), ("stride_vcb",.integer), ("stride_vch",.integer), ("stride_vcs",.integer), ("stride_vcd",.integer), ("stride_bts",.integer), ("stride_btb",.integer), ("block_size",.integer)]
  constexpr := ["HEAD_DIM", "KCACHE_X"]
  readPorts := ["context_lengths", "BLOCK_TABLES", "K", "V"]
  writePorts := ["KCache", "VCache"]

def sourceKernel (K V KCache VCache BLOCK_TABLES context_lengths : RegionName)
    (stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_kcd stride_vcb stride_vch stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat) : ComputeKernel :=
  with_triton_call callSignature call_args [stride_kt := stride_kt, stride_kh := stride_kh, stride_kd := stride_kd, stride_vt := stride_vt, stride_vh := stride_vh, stride_vd := stride_vd, stride_kcb := stride_kcb, stride_kch := stride_kch, stride_kcsplit_x := stride_kcsplit_x, stride_kcs := stride_kcs, stride_kcd := stride_kcd, stride_vcb := stride_vcb, stride_vch := stride_vch, stride_vcs := stride_vcs, stride_vcd := stride_vcd, stride_bts := stride_bts, stride_btb := stride_btb, block_size := block_size] in triton {
    cur_seq_idx = tl.program_id(0)
    cur_kv_head_idx = tl.program_id(1)

    past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - 1
    last_bt_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb)
    offsets_in_last_block = past_kv_seq_len % block_size

    range_x = tl.arange(0, KCACHE_X)
    offsets_dmodel_x_partition = tl.arange(0, KCACHE_X)

    for split_x in tl.static_range(HEAD_DIM // KCACHE_X):
        offsets_dmodel_x_partition = tl.arange(split_x * KCACHE_X, (split_x + 1) * KCACHE_X)
        offsets_k = cur_seq_idx * stride_kt + cur_kv_head_idx * stride_kh + offsets_dmodel_x_partition * stride_kd
        k = tl.load(K + offsets_k)
        offsets_v = cur_seq_idx * stride_vt + cur_kv_head_idx * stride_vh + offsets_dmodel_x_partition * stride_vd
        v = tl.load(V + offsets_v)

        offsets_kcache = (
            block_id * stride_kcb
            + cur_kv_head_idx * stride_kch
            + split_x * stride_kcsplit_x
            + offsets_in_last_block * stride_kcs
            + range_x
        )
        tl.store(KCache + offsets_kcache, k)
        offsets_vcache = (
            block_id * stride_vcb
            + cur_kv_head_idx * stride_vch
            + offsets_in_last_block * stride_vcs
            + offsets_dmodel_x_partition * stride_vcd
        )
        tl.store(VCache + offsets_vcache, v)
    return
  }

def commutedSourcePastedKernel (K V KCache VCache BLOCK_TABLES context_lengths : RegionName)
    (stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_kcd stride_vcb stride_vch stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat) : ComputeKernel :=
  with_triton_call callSignature call_args [stride_kt := stride_kt, stride_kh := stride_kh, stride_kd := stride_kd, stride_vt := stride_vt, stride_vh := stride_vh, stride_vd := stride_vd, stride_kcb := stride_kcb, stride_kch := stride_kch, stride_kcsplit_x := stride_kcsplit_x, stride_kcs := stride_kcs, stride_kcd := stride_kcd, stride_vcb := stride_vcb, stride_vch := stride_vch, stride_vcs := stride_vcs, stride_vcd := stride_vcd, stride_bts := stride_bts, stride_btb := stride_btb, block_size := block_size] in triton {
    cur_seq_idx = tl.program_id(0)
    cur_kv_head_idx = tl.program_id(1)

    past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - 1
    last_bt_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb)
    offsets_in_last_block = past_kv_seq_len % block_size

    range_x = tl.arange(0, KCACHE_X)
    offsets_dmodel_x_partition = tl.arange(0, KCACHE_X)

    for split_x in tl.static_range(HEAD_DIM // KCACHE_X):
        offsets_dmodel_x_partition = tl.arange(split_x * KCACHE_X, (split_x + 1) * KCACHE_X)
        offsets_k = cur_kv_head_idx * stride_kh + cur_seq_idx * stride_kt + offsets_dmodel_x_partition * stride_kd
        k = tl.load(K + offsets_k)
        offsets_v = cur_seq_idx * stride_vt + cur_kv_head_idx * stride_vh + offsets_dmodel_x_partition * stride_vd
        v = tl.load(V + offsets_v)

        offsets_kcache = (
            block_id * stride_kcb
            + cur_kv_head_idx * stride_kch
            + split_x * stride_kcsplit_x
            + offsets_in_last_block * stride_kcs
            + range_x
        )
        tl.store(KCache + offsets_kcache, k)
        offsets_vcache = (
            block_id * stride_vcb
            + cur_kv_head_idx * stride_vch
            + offsets_in_last_block * stride_vcs
            + offsets_dmodel_x_partition * stride_vcd
        )
        tl.store(VCache + offsets_vcache, v)
    return
  }

def quotedOriginal (p : Parameters) : ComputeKernel :=
  sourceKernel "key" "value" "keyCache" "valueCache" "table" "lengths"
    p.strideKt p.strideKh p.strideKd p.strideVt p.strideVh p.strideVd p.strideKcb p.strideKch p.strideKcsplit p.strideKcs 0 p.strideVcb p.strideVch p.strideVcs p.strideVcd p.strideBts p.strideBtb p.blockSize p.dims p.width

def quotedCommuted (p : Parameters) : ComputeKernel :=
  commutedSourcePastedKernel "key" "value" "keyCache" "valueCache" "table" "lengths"
    p.strideKt p.strideKh p.strideKd p.strideVt p.strideVh p.strideVd p.strideKcb p.strideKch p.strideKcsplit p.strideKcs 0 p.strideVcb p.strideVch p.strideVcs p.strideVcd p.strideBts p.strideBtb p.blockSize p.dims p.width

def kernel (commuted : Bool) : Parameters → ComputeKernel :=
  match commuted with
  | Bool.false => quotedOriginal
  | Bool.true => quotedCommuted

def originalKernel := kernel Bool.false
def commutedKernel := kernel Bool.true

end VeriTile.Bench.Repairs.KvCacheCopy.Source
