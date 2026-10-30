import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.DSL

/-! Canonical original FP32 K/V copy. INT32 metadata and source index arithmetic
are preserved. The static loop covers every split; the range translation
`arange(q*width,(q+1)*width) = q*width + arange(0,width)` is explicit below.
The candidate commutes only the two scalar K-source offsets. -/
namespace VeriTile.Bench.Repairs.KvCacheCopy.Source
open VeriTile.Triton

structure Parameters where
  (strideKt strideKh strideKd strideVt strideVh strideVd
    strideKcb strideKch strideKcsplit strideKcs
    strideVcb strideVch strideVcs strideVcd strideBts strideBtb
    blockSize dims width : Nat)

/-- Actual selected source ABI; all runtime and unused arguments are retained. -/
def sourceSignature : CallSignature where
  pointers := [("K",.fp32),("V",.fp32),("KCache",.fp32),("VCache",.fp32),
    ("BLOCK_TABLES",.i32),("context_lengths",.i32)]
  scalars := [("stride_kt",.integer), ("stride_kh",.integer), ("stride_kd",.integer), ("stride_vt",.integer), ("stride_vh",.integer), ("stride_vd",.integer), ("stride_kcb",.integer), ("stride_kch",.integer), ("stride_kcsplit_x",.integer), ("stride_kcs",.integer), ("stride_kcd",.integer), ("stride_vcb",.integer), ("stride_vch",.integer), ("stride_vcs",.integer), ("stride_vcd",.integer), ("stride_bts",.integer), ("stride_btb",.integer), ("block_size",.integer)]
  constexpr := ["HEAD_DIM","KCACHE_X"]
  readPorts := ["K","V","BLOCK_TABLES","context_lengths"]
  writePorts := ["KCache","VCache"]

/-- Complete selected Python body; stride_kcd is unused, with a compatibility
call default only. The actual body and all its parameter roles are preserved. -/
def quotedOriginal (p : Parameters) (stride_kcd : Nat := 1) : ComputeKernel :=
  let K : RegionName := "key"
  let V : RegionName := "value"
  let KCache : RegionName := "keyCache"
  let VCache : RegionName := "valueCache"
  let BLOCK_TABLES : Region .int := ⟨"table"⟩
  let context_lengths : Region .int := ⟨"lengths"⟩
  let HEAD_DIM := p.dims
  let KCACHE_X := p.width
  with_triton_call sourceSignature call_args [stride_kt := p.strideKt, stride_kh := p.strideKh, stride_kd := p.strideKd, stride_vt := p.strideVt, stride_vh := p.strideVh, stride_vd := p.strideVd, stride_kcb := p.strideKcb, stride_kch := p.strideKch, stride_kcsplit_x := p.strideKcsplit, stride_kcs := p.strideKcs, stride_kcd := stride_kcd, stride_vcb := p.strideVcb, stride_vch := p.strideVch, stride_vcs := p.strideVcs, stride_vcd := p.strideVcd, stride_bts := p.strideBts, stride_btb := p.strideBtb, block_size := p.blockSize] in
  triton {
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

/-- Complete selected Python body; stride_kcd is unused, with a compatibility
call default only. The actual body and all its parameter roles are preserved. -/
def quotedCommuted (p : Parameters) (stride_kcd : Nat := 1) : ComputeKernel :=
  let K : RegionName := "key"
  let V : RegionName := "value"
  let KCache : RegionName := "keyCache"
  let VCache : RegionName := "valueCache"
  let BLOCK_TABLES : Region .int := ⟨"table"⟩
  let context_lengths : Region .int := ⟨"lengths"⟩
  let HEAD_DIM := p.dims
  let KCACHE_X := p.width
  with_triton_call sourceSignature call_args [stride_kt := p.strideKt, stride_kh := p.strideKh, stride_kd := p.strideKd, stride_vt := p.strideVt, stride_vh := p.strideVh, stride_vd := p.strideVd, stride_kcb := p.strideKcb, stride_kch := p.strideKch, stride_kcsplit_x := p.strideKcsplit, stride_kcs := p.strideKcs, stride_kcd := stride_kcd, stride_vcb := p.strideVcb, stride_vch := p.strideVch, stride_vcs := p.strideVcs, stride_vcd := p.strideVcd, stride_bts := p.strideBts, stride_btb := p.strideBtb, block_size := p.blockSize] in
  triton {
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

def kernel (commuted : Bool) (p : Parameters) : ComputeKernel :=
  if commuted then quotedCommuted p else quotedOriginal p

def originalKernel := kernel Bool.false
def commutedKernel := kernel Bool.true

end VeriTile.Bench.Repairs.KvCacheCopy.Source
