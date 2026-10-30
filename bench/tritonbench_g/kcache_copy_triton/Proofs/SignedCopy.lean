import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Integer

/-! Complete multi-token signed copy. The source is the existing signed
transcription with explicit floating storage annotations. All original token,
sequence and split-x program coordinates are retained. Arithmetic follows the
source model's unbounded signed metadata and Nat addresses; machine-width
integer refinement is a separate obligation, never a numerical assumption. -/
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.SignedCopy
open Lean VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

open TypedCopy (Storage copyBackend scatter_memory)

structure Parameters where
  (K KCache : RegionName)
  (blockTables lengths : Region .int)
  (nTokens strideKt strideKh strideKd strideKcb strideKch strideKcsplit
    strideKcs strideBts strideBtb blockSize width : Nat)

syntax "kcacheSignedSource%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(kcacheSignedSource%[$storage:tritonDType]) =>
    `(fun (p : Parameters) =>
      let K := p.K
      let KCache := p.KCache
      let BLOCK_TABLES := p.blockTables
      let seq_lengths := p.lengths
      ComputeKernel.mk [K, BLOCK_TABLES.cast, seq_lengths.cast] [KCache]
      (triton {
        tl.region K = $storage, KCache = $storage
        cur_token_idx_nat = tl.program_id(0)
        cur_token_idx = tl.cast(cur_token_idx_nat, tl.int64)
        cur_seq_idx = cur_token_idx_nat // $(p.nTokens)
        cur_seq_idx_i = tl.cast(cur_seq_idx, tl.int64)
        n_tokens_i = tl.cast($(p.nTokens), tl.int64)
        one_i = tl.cast($(1), tl.int64)
        cur_token_shift = cur_token_idx - n_tokens_i * (cur_seq_idx_i + one_i)
        cur_kv_head_idx = tl.program_id(1)
        split_x_idx = tl.program_id(2)
        past_kv_seq_len = tl.load($((seq_lengths : Region TileDType.int)) + cur_seq_idx) + cur_token_shift
        last_bt_block_idx = past_kv_seq_len // $(p.blockSize)
        last_bt_block_idx_nat = tl.cast(last_bt_block_idx, tl.uint64)
        block_table_ptr = $((BLOCK_TABLES : Region TileDType.int)) + cur_seq_idx * $(p.strideBts)
        block_id = tl.load(block_table_ptr + last_bt_block_idx_nat * $(p.strideBtb))
        offset_last_block = past_kv_seq_len % $(p.blockSize)
        block_id_nat = tl.cast(block_id, tl.uint64)
        offset_last_block_nat = tl.cast(offset_last_block, tl.uint64)
        offsets_dmodel = split_x_idx * $(p.width) + tl.arange(0, $(p.width))
        offsets_k = cur_token_idx_nat * $(p.strideKt) +
          cur_kv_head_idx * $(p.strideKh) + offsets_dmodel * $(p.strideKd)
        k = tl.load(K + offsets_k)
        offsets_kcache = block_id_nat * $(p.strideKcb) +
          cur_kv_head_idx * $(p.strideKch) + split_x_idx * $(p.strideKcsplit) +
          offset_last_block_nat * $(p.strideKcs) + tl.arange(0, $(p.width))
        tl.store(KCache + offsets_kcache, k)
      }).surfaceBody)

def source : Storage → Parameters → ComputeKernel
  | .fp16 => kcacheSignedSource%[tl.float16]
  | .bf16 => kcacheSignedSource%[tl.bfloat16]
  | .fp32 => kcacheSignedSource%[tl.float32]
  | .fp64 => kcacheSignedSource%[tl.float64]
  | .f8e4 => kcacheSignedSource%[tl.float8e4nv]
  | .f8e5 => kcacheSignedSource%[tl.float8e5]

def sequence (p : Parameters) (ids : Nat → Nat) : Nat := ids 0 / p.nTokens

def tokenShift (p : Parameters) (ids : Nat → Nat) : Int :=
  (ids 0 : Int) - (p.nTokens : Int) * ((sequence p ids : Int) + 1)

def past {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Int :=
  (s.mem p.lengths.cast (sequence p s.pids)).read .int + tokenShift p s.pids

def unsigned64 (x : Int) : Nat := (x % 2^64).toNat

def blockId {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Int :=
  (s.mem p.blockTables.cast (sequence p s.pids * p.strideBts +
    unsigned64 ((past p s).tdiv (p.blockSize : Int)) * p.strideBtb)).read .int

def inputOffset {α : Type} (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Nat :=
  s.pids 0 * p.strideKt + s.pids 1 * p.strideKh +
    (s.pids 2 * p.width + i.1.val) * p.strideKd

def outputOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Nat :=
  unsigned64 (blockId p s) * p.strideKcb + s.pids 1 * p.strideKch +
    s.pids 2 * p.strideKcsplit + unsigned64 ((past p s).tmod (p.blockSize : Int)) * p.strideKcs + i.1.val

def loaded {α : Type} [Inhabited α] (d : Storage) (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Cell α :=
  .mk d.tag ((s.mem p.K (inputOffset p s i)).read d.tag)

def copied {α : Type} [Inhabited α] (d : Storage) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl
    (fun t i => t.write p.KCache (outputOffset p s i) (loaded d p s i)) s

/-- Successful execution of all selected source statements, from arbitrary
initial registers. No floating algebraic law or finite-value condition. -/
def execution {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec (copyBackend rest) (source d p) s = some t} := by
  cases d <;> (refine ⟨?_, ?_⟩
               swap
               simp [source, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.numericLt, copyBackend, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind]
               try rfl)


/-- This is the entire previously registered signed source, after forgetting
floating storage annotations. No metadata computation, signed shift or store
is removed; all three program coordinates are the original coordinates. -/
theorem source_projection (d : Storage) (p : Parameters) :
    (source d p).eraseDType.toAlgKernel.body =
      (copy_to_kcache_seqlen_n_signed_kernel p.K p.KCache p.blockTables p.lengths
        p.strideKt p.strideKh p.strideKd p.strideKcb p.strideKch
        p.strideKcsplit p.strideKcs 1 p.strideBts p.strideBtb p.blockSize
        p.nTokens 0 p.width).toAlgKernel.body := by
  cases d
  all_goals simp only [source, copy_to_kcache_seqlen_n_signed_kernel,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast,
    Except.bind, bind]
  all_goals rfl

/-- Exact operational memory effect, including source/output aliasing. The
right side is an independent address-indexed specification of the copy. -/
theorem execution_memory {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) :
    (execution rest d p s).val.mem = (copied d p s).mem := by
  cases d <;> simp [execution, copied, scatter_memory, loaded, Storage.tag,
    inputOffset, outputOffset, blockId, past, sequence, tokenShift, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, integralFloorDiv, integralMod,
    unsigned64, ← IntegerExpr.tdiv_formula, Int.tmod_def, Region.cast]

 theorem output_injective {α : Type} [Inhabited α] (p : Parameters) (s : State α) :
    Function.Injective (outputOffset p s) := by
  intro i j h
  have hv : i.1.val = j.1.val := Nat.add_left_cancel h
  exact Prod.ext (Fin.ext hv) (Subsingleton.elim _ _)

/-- Every lane is read from the original typed source and written to its
independent cache address. NaNs, infinities and signed zeros are not excluded. -/
theorem readback {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) :
    (execution rest d p s).val.mem p.KCache (outputOffset p s i) =
      .mk d.tag ((s.mem p.K (inputOffset p s i)).read d.tag) := by
  rw [execution_memory]
  exact State.scatter_readback _ _ _ _ i (output_injective p s)


/-- Well-typed input cells are copied byte-for-byte in an opaque bit carrier.
The carrier may contain every bit pattern, including signaling NaN payloads. -/
theorem typed_readback {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (value : Value α d.tag)
    (htyped : s.mem p.K (inputOffset p s i) = .mk d.tag value) :
    (execution rest d p s).val.mem p.KCache (outputOffset p s i) = .mk d.tag value := by
  rw [readback, htyped, Cell.read_same]

/-- Cells outside the selected destination tile retain their complete tagged
payload; this includes other regions and holes before/after the tile. -/
theorem frame {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat)
    (hmiss : r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], o ≠ outputOffset p s i) :
    (execution rest d p s).val.mem r o = s.mem r o := by
  rw [execution_memory]
  apply State.scatter_frame
  exact hmiss.imp id (fun h i _ => h i)

/-- A public contract packages success, independent readback and framing.
There is no numerical side condition and no same-program equality. -/
theorem correctness {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec (copyBackend rest) (source d p) s = some t ∧
      (∀ i : TileIndex [p.width], t.mem p.KCache (outputOffset p s i) =
        .mk d.tag ((s.mem p.K (inputOffset p s i)).read d.tag)) ∧
      (∀ r o, (r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], o ≠ outputOffset p s i) →
        t.mem r o = s.mem r o) :=
  ⟨(execution rest d p s).val, (execution rest d p s).property,
    readback rest d p s, frame rest d p s⟩

#axiomsClean source_projection
#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean readback
#axiomsClean frame
#axiomsClean correctness

end
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.SignedCopy
