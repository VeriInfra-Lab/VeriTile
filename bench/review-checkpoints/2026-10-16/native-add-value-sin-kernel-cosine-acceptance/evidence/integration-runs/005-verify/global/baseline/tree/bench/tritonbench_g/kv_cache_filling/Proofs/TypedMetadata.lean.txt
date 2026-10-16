import bench.tritonbench_g.kv_cache_filling.KvCacheFilling
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

/-! Typed scale/zero metadata writeback stage. `MetaPre` is an input buffer,
not an assumed output. Quantizer/reduction producers and fp32-to-uint8 value
stores are outside this theorem. The latter are present even in the supplied
quant_policy=0 test and remain an explicit missing typed conversion obligation. -/
namespace VeriTile.Bench.TritonBenchG.KvCacheFilling.TypedMetadata
open Lean VeriTile.Triton FP.Structural KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure Parameters where
  (input output : RegionName)
  blockOffsets : Region .nat
  (blockIndex kvBlockIndex slot strideMn strideMb strideMh strideMd strideBoff heads width : Nat)

syntax "kvMetaTypedSource%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(kvMetaTypedSource%[$storage:tritonDType]) =>
    `(fun (p : Parameters) =>
      let MetaPre := p.input
      let MetaOut := p.output
      let BlockOffsets := p.blockOffsets
      ComputeKernel.mk [MetaPre, BlockOffsets.cast] [MetaOut]
      (triton {
        tl.region MetaPre = $storage, MetaOut = $storage
        batch_id = tl.program_id(0)
        h_off = tl.arange(0, $(p.width))
        block_off = tl.load(BlockOffsets + batch_id * $(p.strideBoff) + $(p.kvBlockIndex))
        mask = h_off < $(p.heads)
        meta_val = tl.load(MetaPre + h_off, mask=mask, other=0.0)
        tl.store(MetaOut + block_off * $(p.strideMn) + $(p.blockIndex) * $(p.strideMb) +
          h_off * $(p.strideMh) + $(p.slot) * $(p.strideMd), meta_val, mask=mask)
      }).surfaceBody)

def source : Storage → Parameters → ComputeKernel
  | .fp16 => kvMetaTypedSource%[tl.float16]
  | .bf16 => kvMetaTypedSource%[tl.bfloat16]
  | .fp32 => kvMetaTypedSource%[tl.float32]
  | .fp64 => kvMetaTypedSource%[tl.float64]
  | .f8e4 => kvMetaTypedSource%[tl.float8e4nv]
  | .f8e5 => kvMetaTypedSource%[tl.float8e5]

def active (p : Parameters) (i : TileIndex [p.width]) : Prop := i.1.val < p.heads
instance (p : Parameters) : DecidablePred (active p) := fun _ => inferInstanceAs (Decidable (_ < _))

def blockOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem p.blockOffsets.cast (s.pids 0 * p.strideBoff + p.kvBlockIndex)).read .nat

def outputOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Nat :=
  blockOffset p s * p.strideMn + p.blockIndex * p.strideMb +
    i.1.val * p.strideMh + p.slot * p.strideMd

def loaded {α : Type} [Inhabited α] (d : Storage) (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Cell α :=
  .mk d.tag ((s.mem p.input i.1.val).read d.tag)

def copied {α : Type} [Inhabited α] (d : Storage) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl (fun t i =>
    if active p i then t.write p.output (outputOffset p s i) (loaded d p s i) else t) s

def execution {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec (copyBackend rest) (source d p) s = some t} := by
  cases d <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [source, FP.Structural.exec, ComputeKernel.surfaceBody,
       FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
       FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
       FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
       FP.Structural.bop, FP.Structural.natLt, copyBackend, ComputeDType.eraseDType,
       FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind]
     try rfl)

theorem source_projection (d : Storage) (p : Parameters) :
    (source d p).eraseDType.toAlgKernel.body =
      (fill_quant_meta_store_slice p.input p.output p.blockOffsets p.blockIndex p.kvBlockIndex
        p.slot p.strideMn p.strideMb p.strideMh p.strideMd p.strideBoff p.heads p.width).toAlgKernel.body := by
  cases d
  all_goals simp only [source, fill_quant_meta_store_slice,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  all_goals rfl

theorem execution_memory {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) :
    (execution rest d p s).val.mem = (copied d p s).mem := by
  cases d
  all_goals simp +contextual [execution, copied, masked_scatter_memory, loaded,
    Storage.tag, active, outputOffset, blockOffset, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]

theorem readback {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (hi : active p i)
    (hinj : ∀ a b, active p a → active p b → outputOffset p s a = outputOffset p s b → a = b) :
    (execution rest d p s).val.mem p.output (outputOffset p s i) =
      .mk d.tag ((s.mem p.input i.1.val).read d.tag) := by
  rw [execution_memory]
  exact masked_readback_active _ _ _ _ _ i hi hinj


theorem typed_readback {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (hi : active p i)
    (hinj : ∀ a b, active p a → active p b → outputOffset p s a = outputOffset p s b → a = b)
    (value : Value α d.tag) (htyped : s.mem p.input i.1.val = .mk d.tag value) :
    (execution rest d p s).val.mem p.output (outputOffset p s i) = .mk d.tag value := by
  rw [readback rest d p s i hi hinj, htyped, Cell.read_same]

theorem frame {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat)
    (hmiss : r ≠ p.output ∨ ∀ i : TileIndex [p.width], active p i → o ≠ outputOffset p s i) :
    (execution rest d p s).val.mem r o = s.mem r o := by
  rw [execution_memory]
  apply State.masked_scatter_frame
  exact hmiss.imp id (fun h i _ => h i)

theorem correctness {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α)
    (hinj : ∀ a b, active p a → active p b → outputOffset p s a = outputOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (copyBackend rest) (source d p) s = some t ∧
      (∀ i : TileIndex [p.width], active p i → t.mem p.output (outputOffset p s i) =
        .mk d.tag ((s.mem p.input i.1.val).read d.tag)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ i : TileIndex [p.width], active p i → o ≠ outputOffset p s i) →
        t.mem r o = s.mem r o) :=
  ⟨(execution rest d p s).val, (execution rest d p s).property,
    fun i hi => readback rest d p s i hi hinj, frame rest d p s⟩

#axiomsClean source_projection
#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean readback
#axiomsClean typed_readback
#axiomsClean frame
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.KvCacheFilling.TypedMetadata
