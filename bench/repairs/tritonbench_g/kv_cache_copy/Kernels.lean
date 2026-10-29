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

/-- Complete ordinary original and existing offset-commuted source programs. -/
def quotedOriginal :=
  fun (p : Parameters) =>
    let K : RegionName := "key"
    let V : RegionName := "value"
    let KCache : RegionName := "keyCache"
    let VCache : RegionName := "valueCache"
    let BLOCK_TABLES : Region .int := ⟨"table"⟩
    let context_lengths : Region .int := ⟨"lengths"⟩
    triton {
      tl.region K = tl.float32, V = tl.float32,
        KCache = tl.float32, VCache = tl.float32,
        BLOCK_TABLES = tl.int32, context_lengths = tl.int32
      cur_seq_idx = tl.program_id(0)
      cur_kv_head_idx = tl.program_id(1)
      past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - $(1)
      last_bt_block_idx = past_kv_seq_len // $(p.blockSize)
      block_table_ptr = BLOCK_TABLES + cur_seq_idx * $(p.strideBts)
      block_id = tl.load(block_table_ptr + last_bt_block_idx * $(p.strideBtb))
      offsets_in_last_block = past_kv_seq_len % $(p.blockSize)
      range_x = tl.arange(0, $(p.width))
      offsets_dmodel_x_partition = tl.arange(0, $(p.width))
      for split_x in tl.static_range($(p.dims / p.width)) {
        offsets_dmodel_x_partition = split_x * $(p.width) + tl.arange(0, $(p.width))
        offsets_k = cur_seq_idx * $(p.strideKt) +
          cur_kv_head_idx * $(p.strideKh) + offsets_dmodel_x_partition * $(p.strideKd)
        k = tl.load(K + offsets_k)
        offsets_v = cur_seq_idx * $(p.strideVt) + cur_kv_head_idx * $(p.strideVh) +
          offsets_dmodel_x_partition * $(p.strideVd)
        v = tl.load(V + offsets_v)
        offsets_kcache = block_id * $(p.strideKcb) + cur_kv_head_idx * $(p.strideKch) +
          split_x * $(p.strideKcsplit) + offsets_in_last_block * $(p.strideKcs) + range_x
        tl.store(KCache + offsets_kcache, k)
        offsets_vcache = block_id * $(p.strideVcb) + cur_kv_head_idx * $(p.strideVch) +
          offsets_in_last_block * $(p.strideVcs) + offsets_dmodel_x_partition * $(p.strideVcd)
        tl.store(VCache + offsets_vcache, v)
      }
    }

def quotedCommuted :=
  fun (p : Parameters) =>
    let K : RegionName := "key"
    let V : RegionName := "value"
    let KCache : RegionName := "keyCache"
    let VCache : RegionName := "valueCache"
    let BLOCK_TABLES : Region .int := ⟨"table"⟩
    let context_lengths : Region .int := ⟨"lengths"⟩
    triton {
      tl.region K = tl.float32, V = tl.float32,
        KCache = tl.float32, VCache = tl.float32,
        BLOCK_TABLES = tl.int32, context_lengths = tl.int32
      cur_seq_idx = tl.program_id(0)
      cur_kv_head_idx = tl.program_id(1)
      past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - $(1)
      last_bt_block_idx = past_kv_seq_len // $(p.blockSize)
      block_table_ptr = BLOCK_TABLES + cur_seq_idx * $(p.strideBts)
      block_id = tl.load(block_table_ptr + last_bt_block_idx * $(p.strideBtb))
      offsets_in_last_block = past_kv_seq_len % $(p.blockSize)
      range_x = tl.arange(0, $(p.width))
      offsets_dmodel_x_partition = tl.arange(0, $(p.width))
      for split_x in tl.static_range($(p.dims / p.width)) {
        offsets_dmodel_x_partition = split_x * $(p.width) + tl.arange(0, $(p.width))
        offsets_k = cur_kv_head_idx * $(p.strideKh) +
          cur_seq_idx * $(p.strideKt) + offsets_dmodel_x_partition * $(p.strideKd)
        k = tl.load(K + offsets_k)
        offsets_v = cur_seq_idx * $(p.strideVt) + cur_kv_head_idx * $(p.strideVh) +
          offsets_dmodel_x_partition * $(p.strideVd)
        v = tl.load(V + offsets_v)
        offsets_kcache = block_id * $(p.strideKcb) + cur_kv_head_idx * $(p.strideKch) +
          split_x * $(p.strideKcsplit) + offsets_in_last_block * $(p.strideKcs) + range_x
        tl.store(KCache + offsets_kcache, k)
        offsets_vcache = block_id * $(p.strideVcb) + cur_kv_head_idx * $(p.strideVch) +
          offsets_in_last_block * $(p.strideVcs) + offsets_dmodel_x_partition * $(p.strideVcd)
        tl.store(VCache + offsets_vcache, v)
      }
    }

def kernel (commuted : Bool) : Parameters → ComputeKernel :=
  match commuted with
  | Bool.false => quotedOriginal
  | Bool.true => quotedCommuted

def originalKernel := kernel Bool.false
def commutedKernel := kernel Bool.true

end VeriTile.Bench.Repairs.KvCacheCopy.Source
