import bench.tritonbench_g.kcache_copy_triton.KcacheCopyTriton
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit

/-! Typed execution of the already selected n_tokens = 1, single split-x
source. Floating payloads are opaque; the model never uses real arithmetic.
The fp32 surface uses a precision-tagged load, whose raw-memory backend is
identity. All other numerical callbacks are arbitrary and unused. Metadata
retains the existing Nat address model, including its existing no-overflow
abstraction. This is not a proof of the unselected n_tokens > 1 branch. -/
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
open Lean VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

inductive Storage where
  | fp16 | bf16 | fp32 | fp64 | f8e4 | f8e5
  deriving DecidableEq, Repr

def Storage.tag : Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 | .fp64 => .real
  | .f8e4 => .f8e4 | .f8e5 => .f8e5

structure Parameters where
  (K KCache : RegionName)
  (blockTables lengths : Region .nat)
  (splitX strideKt strideKh strideKd strideKcb strideKch strideKcsplit
    strideKcs strideBts strideBtb blockSize width : Nat)

syntax "kcacheTypedSource%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(kcacheTypedSource%[$storage:tritonDType]) =>
    `(fun (p : Parameters) =>
      let K := p.K
      let KCache := p.KCache
      let BLOCK_TABLES := p.blockTables
      let seq_lengths := p.lengths
      ComputeKernel.mk [K, BLOCK_TABLES.cast, seq_lengths.cast] [KCache]
      (triton {
        tl.region K = $storage, KCache = $storage
        cur_token_idx = tl.program_id(0)
        cur_seq_idx = cur_token_idx
        cur_kv_head_idx = tl.program_id(1)
        past_kv_seq_len = tl.load(seq_lengths + cur_seq_idx) - $(1)
        last_bt_block_idx = past_kv_seq_len // $(p.blockSize)
        block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(p.strideBts) +
          last_bt_block_idx * $(p.strideBtb))
        offset_last_block = past_kv_seq_len % $(p.blockSize)
        offsets_dmodel = $(p.splitX) * $(p.width) + tl.arange(0, $(p.width))
        k = tl.load(K + cur_token_idx * $(p.strideKt) +
          cur_kv_head_idx * $(p.strideKh) + offsets_dmodel * $(p.strideKd))
        tl.store(KCache + block_id * $(p.strideKcb) +
          cur_kv_head_idx * $(p.strideKch) + $(p.splitX) * $(p.strideKcsplit) +
          offset_last_block * $(p.strideKcs) + tl.arange(0, $(p.width)), k)
      }).surfaceBody)

def source : Storage → Parameters → ComputeKernel
  | .fp16 => kcacheTypedSource%[tl.float16]
  | .bf16 => kcacheTypedSource%[tl.bfloat16]
  | .fp32 => kcacheTypedSource%[tl.float32]
  | .fp64 => kcacheTypedSource%[tl.float64]
  | .f8e4 => kcacheTypedSource%[tl.float8e4nv]
  | .f8e5 => kcacheTypedSource%[tl.float8e5]

/-- Raw homogeneous copy: the explicit fp32 memory read preserves its payload.
This is a defined load backend, not a universal rounding identity assumption. -/
def copyBackend {α : Type} (rest : Algebra α) : Algebra α :=
  { rest with fp32Load := id }

def past {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem p.lengths.cast (s.pids 0)).read .nat - 1

def blockId {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem p.blockTables.cast
    (s.pids 0 * p.strideBts + past p s / p.blockSize * p.strideBtb)).read .nat

def inputOffset {α : Type} (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Nat :=
  s.pids 0 * p.strideKt + s.pids 1 * p.strideKh +
    (p.splitX * p.width + i.1.val) * p.strideKd

def outputOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Nat :=
  blockId p s * p.strideKcb + s.pids 1 * p.strideKch +
    p.splitX * p.strideKcsplit + past p s % p.blockSize * p.strideKcs + i.1.val

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
    FP.Structural.bop, copyBackend, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind]
               try rfl)


/-- Forgetting storage tags gives exactly the previously selected n1 source
body. No address, metadata lookup, or source statement is dropped. -/
theorem source_projection (d : Storage) (p : Parameters) :
    (source d p).eraseDType.toAlgKernel.body =
      (copy_to_kcache_seqlen_n1_surface p.K p.KCache p.blockTables p.lengths
        p.splitX p.strideKt p.strideKh p.strideKd p.strideKcb p.strideKch
        p.strideKcsplit p.strideKcs 1 p.strideBts p.strideBtb p.blockSize
        1 0 p.width).toAlgKernel.body := by
  cases d
  all_goals simp only [source, copy_to_kcache_seqlen_n1_surface,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast,
    Except.bind, bind]
  all_goals rfl

theorem scatter_memory {α ι : Type} (xs : List ι) (region : RegionName)
    (offset : ι → Nat) (value : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => t.write region (offset i) (value i)) s).mem =
      xs.foldl (fun mem i => fun r o =>
        if r = region ∧ o = offset i then value i else mem r o) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih => simp only [List.foldl_cons, ih]; rfl


@[simp] theorem masked_scatter_regs {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (region : RegionName)
    (offset : ι → Nat) (value : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write region (offset i) (value i)
      else t) s).regs = s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    split <;> rfl

theorem masked_scatter_memory {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (region : RegionName)
    (offset : ι → Nat) (value : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write region (offset i) (value i)
      else t) s).mem =
      xs.foldl (fun mem i => if active i then fun r o =>
        if r = region ∧ o = offset i then value i else mem r o else mem) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    by_cases hi : active i <;> simp only [hi, ↓reduceIte] <;> rfl


/-- Readback needs injectivity only among active lanes, allowing arbitrary
inactive offsets and the one-active-lane case even with a zero stride. -/
theorem masked_readback_active {α : Type} (r : RegionName)
    (off : TileIndex shape → Nat) (val : TileIndex shape → Cell α)
    (active : TileIndex shape → Prop) [DecidablePred active] (s : State α)
    (i : TileIndex shape) (hi : active i)
    (hinj : ∀ a b, active a → active b → off a = off b → a = b) :
    ((TileShape.allIndices shape).foldl
      (fun t k => if active k then t.write r (off k) (val k) else t) s).mem r (off i) = val i := by
  rw [State.masked_scatter_eq_filter]
  have himem : i ∈ (TileShape.allIndices shape).filter (fun j => decide (active j)) := by
    simp [TileShape.mem_allIndices, hi]
  obtain ⟨before, after, hl⟩ := List.append_of_mem himem
  have hnd := (TileShape.allIndices_nodup shape).filter (fun j => decide (active j))
  rw [hl, List.nodup_append, List.nodup_cons] at hnd
  obtain ⟨_, ⟨hnot, _⟩, _⟩ := hnd
  rw [hl, List.foldl_append, List.foldl_cons, State.scatter_frame]
  · exact State.write_same _ _ _ _
  · right
    intro j hj heq
    have hjmem : j ∈ (TileShape.allIndices shape).filter (fun j => decide (active j)) := by
      rw [hl]
      simp only [List.mem_append, List.mem_cons]
      exact Or.inr (Or.inr hj)
    have hjactive : active j := by simpa using (List.mem_filter.mp hjmem).2
    exact hnot ((hinj i j hi hjactive heq) ▸ hj)

/-- Exact operational memory effect, including source/output aliasing. The
right side is an independent address-indexed specification of the copy. -/
theorem execution_memory {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) :
    (execution rest d p s).val.mem = (copied d p s).mem := by
  cases d <;> simp [execution, copied, scatter_memory, loaded, Storage.tag,
    inputOffset, outputOffset, blockId, past, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, integralFloorDiv, integralMod, Region.cast]

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
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
