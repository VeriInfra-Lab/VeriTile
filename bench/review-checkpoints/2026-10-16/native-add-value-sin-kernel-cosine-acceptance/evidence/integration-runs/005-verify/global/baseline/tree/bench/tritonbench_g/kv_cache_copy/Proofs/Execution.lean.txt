import bench.tritonbench_g.kv_cache_copy.KvCacheCopy
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

/-! Raw typed payload execution of the accepted single x-block decode slice.
The K and V storage dtypes vary independently; each source/cache pair is
homogeneous. Both loads precede both stores. Address metadata retains the
accepted Nat semantics; no claim about machine-address overflow is added. -/
namespace VeriTile.Bench.TritonBenchG.KvCacheCopy.TypedCopy
open Lean VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

structure Parameters extends KcacheCopyTriton.TypedCopy.Parameters where
  (V VCache : RegionName)
  (strideVt strideVh strideVd strideVcb strideVch strideVcs strideVcd headDim : Nat)

syntax "kvCacheTypedSource%" "[" tritonDType "," tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(kvCacheTypedSource%[$ktype:tritonDType, $vtype:tritonDType]) =>
    `(fun (p : Parameters) =>
      let K := p.K
      let KCache := p.KCache
      let V := p.V
      let VCache := p.VCache
      let BLOCK_TABLES := p.blockTables
      let context_lengths := p.lengths
      ComputeKernel.mk [K, V, BLOCK_TABLES.cast, context_lengths.cast] [KCache, VCache]
      (triton {
        tl.region K = $ktype, KCache = $ktype, V = $vtype, VCache = $vtype
        cur_seq_idx = tl.program_id(0)
        cur_kv_head_idx = tl.program_id(1)
        past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - $(1)
        last_bt_block_idx = past_kv_seq_len // $(p.blockSize)
        block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(p.strideBts) +
          last_bt_block_idx * $(p.strideBtb))
        offsets_in_last_block = past_kv_seq_len % $(p.blockSize)
        range_x = tl.arange(0, $(p.width))
        offsets_dmodel_x_partition = $(p.splitX) * $(p.width) + range_x
        k = tl.load(K + cur_seq_idx * $(p.strideKt) +
            cur_kv_head_idx * $(p.strideKh) + offsets_dmodel_x_partition * $(p.strideKd),
          mask=offsets_dmodel_x_partition < $(p.headDim), other=0.0)
        v = tl.load(V + cur_seq_idx * $(p.strideVt) +
            cur_kv_head_idx * $(p.strideVh) + offsets_dmodel_x_partition * $(p.strideVd),
          mask=offsets_dmodel_x_partition < $(p.headDim), other=0.0)
        tl.store(KCache + block_id * $(p.strideKcb) + cur_kv_head_idx * $(p.strideKch) +
            $(p.splitX) * $(p.strideKcsplit) + offsets_in_last_block * $(p.strideKcs) + range_x,
          k, mask=offsets_dmodel_x_partition < $(p.headDim))
        tl.store(VCache + block_id * $(p.strideVcb) + cur_kv_head_idx * $(p.strideVch) +
            offsets_in_last_block * $(p.strideVcs) + offsets_dmodel_x_partition * $(p.strideVcd),
          v, mask=offsets_dmodel_x_partition < $(p.headDim))
      }).surfaceBody)

def source : Storage → Storage → Parameters → ComputeKernel
  | .fp16, .fp16 => kvCacheTypedSource%[tl.float16, tl.float16]
  | .fp16, .bf16 => kvCacheTypedSource%[tl.float16, tl.bfloat16]
  | .fp16, .fp32 => kvCacheTypedSource%[tl.float16, tl.float32]
  | .fp16, .fp64 => kvCacheTypedSource%[tl.float16, tl.float64]
  | .fp16, .f8e4 => kvCacheTypedSource%[tl.float16, tl.float8e4nv]
  | .fp16, .f8e5 => kvCacheTypedSource%[tl.float16, tl.float8e5]
  | .bf16, .fp16 => kvCacheTypedSource%[tl.bfloat16, tl.float16]
  | .bf16, .bf16 => kvCacheTypedSource%[tl.bfloat16, tl.bfloat16]
  | .bf16, .fp32 => kvCacheTypedSource%[tl.bfloat16, tl.float32]
  | .bf16, .fp64 => kvCacheTypedSource%[tl.bfloat16, tl.float64]
  | .bf16, .f8e4 => kvCacheTypedSource%[tl.bfloat16, tl.float8e4nv]
  | .bf16, .f8e5 => kvCacheTypedSource%[tl.bfloat16, tl.float8e5]
  | .fp32, .fp16 => kvCacheTypedSource%[tl.float32, tl.float16]
  | .fp32, .bf16 => kvCacheTypedSource%[tl.float32, tl.bfloat16]
  | .fp32, .fp32 => kvCacheTypedSource%[tl.float32, tl.float32]
  | .fp32, .fp64 => kvCacheTypedSource%[tl.float32, tl.float64]
  | .fp32, .f8e4 => kvCacheTypedSource%[tl.float32, tl.float8e4nv]
  | .fp32, .f8e5 => kvCacheTypedSource%[tl.float32, tl.float8e5]
  | .fp64, .fp16 => kvCacheTypedSource%[tl.float64, tl.float16]
  | .fp64, .bf16 => kvCacheTypedSource%[tl.float64, tl.bfloat16]
  | .fp64, .fp32 => kvCacheTypedSource%[tl.float64, tl.float32]
  | .fp64, .fp64 => kvCacheTypedSource%[tl.float64, tl.float64]
  | .fp64, .f8e4 => kvCacheTypedSource%[tl.float64, tl.float8e4nv]
  | .fp64, .f8e5 => kvCacheTypedSource%[tl.float64, tl.float8e5]
  | .f8e4, .fp16 => kvCacheTypedSource%[tl.float8e4nv, tl.float16]
  | .f8e4, .bf16 => kvCacheTypedSource%[tl.float8e4nv, tl.bfloat16]
  | .f8e4, .fp32 => kvCacheTypedSource%[tl.float8e4nv, tl.float32]
  | .f8e4, .fp64 => kvCacheTypedSource%[tl.float8e4nv, tl.float64]
  | .f8e4, .f8e4 => kvCacheTypedSource%[tl.float8e4nv, tl.float8e4nv]
  | .f8e4, .f8e5 => kvCacheTypedSource%[tl.float8e4nv, tl.float8e5]
  | .f8e5, .fp16 => kvCacheTypedSource%[tl.float8e5, tl.float16]
  | .f8e5, .bf16 => kvCacheTypedSource%[tl.float8e5, tl.bfloat16]
  | .f8e5, .fp32 => kvCacheTypedSource%[tl.float8e5, tl.float32]
  | .f8e5, .fp64 => kvCacheTypedSource%[tl.float8e5, tl.float64]
  | .f8e5, .f8e4 => kvCacheTypedSource%[tl.float8e5, tl.float8e4nv]
  | .f8e5, .f8e5 => kvCacheTypedSource%[tl.float8e5, tl.float8e5]

def active (p : Parameters) (i : TileIndex [p.width]) : Prop :=
  p.splitX * p.width + i.1.val < p.headDim
instance (p : Parameters) : DecidablePred (active p) := fun _ => inferInstanceAs (Decidable (_ < _))

def vInputOffset {α : Type} (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Nat :=
  s.pids 0 * p.strideVt + s.pids 1 * p.strideVh +
    (p.splitX * p.width + i.1.val) * p.strideVd

def vOutputOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Nat :=
  blockId p.toParameters s * p.strideVcb + s.pids 1 * p.strideVch +
    past p.toParameters s % p.blockSize * p.strideVcs +
    (p.splitX * p.width + i.1.val) * p.strideVcd

def vLoaded {α : Type} [Inhabited α] (d : Storage) (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Cell α :=
  .mk d.tag ((s.mem p.V (vInputOffset p s i)).read d.tag)

def copiedK {α : Type} [Inhabited α] (kd : Storage) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl
    (fun t i => if active p i then t.write p.KCache (outputOffset p.toParameters s i)
      (loaded kd p.toParameters s i) else t) s

def copiedKV {α : Type} [Inhabited α] (kd vd : Storage) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl
    (fun t i => if active p i then t.write p.VCache (vOutputOffset p s i)
      (vLoaded vd p s i) else t) (copiedK kd p s)

/-- All twelve source statements succeed from arbitrary initial registers.
The literal used for masked-off loads is arbitrary; no active store observes it. -/
def execution {α : Type} [Inhabited α] (rest : Algebra α) (kd vd : Storage)
    (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec (copyBackend rest) (source kd vd p) s = some t} := by
  cases kd <;> cases vd <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [source, FP.Structural.exec, ComputeKernel.surfaceBody,
       FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
       FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
       FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
       FP.Structural.bop, FP.Structural.natLt, copyBackend, ComputeDType.eraseDType,
       FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind]
     try rfl)


/-- Erasing only storage annotations recovers the exact accepted x-block body. -/
theorem source_projection (kd vd : Storage) (p : Parameters) :
    (source kd vd p).eraseDType.toAlgKernel.body =
      (copy_to_kvcache_seqlen1_xblock p.K p.V p.KCache p.VCache p.blockTables p.lengths
        p.splitX p.strideKt p.strideKh p.strideKd p.strideVt p.strideVh p.strideVd
        p.strideKcb p.strideKch p.strideKcsplit p.strideKcs
        p.strideVcb p.strideVch p.strideVcs p.strideVcd
        p.strideBts p.strideBtb p.blockSize p.headDim p.width).toAlgKernel.body := by
  cases kd <;> cases vd
  all_goals simp only [source, copy_to_kvcache_seqlen1_xblock,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast,
    Except.bind, bind]
  all_goals rfl

#axiomsClean source_projection
#axiomsClean execution
end
end VeriTile.Bench.TritonBenchG.KvCacheCopy.TypedCopy
