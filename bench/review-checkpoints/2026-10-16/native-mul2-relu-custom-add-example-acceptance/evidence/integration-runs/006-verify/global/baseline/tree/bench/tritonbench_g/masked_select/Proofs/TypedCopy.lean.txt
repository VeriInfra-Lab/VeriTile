import bench.tritonbench_g.masked_select.MaskedSelect
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

/-! Actual typed selection-copy source. Prefix sums and selection masks are
inputs to this stage; no prefix-sum producer or expected output is assumed.
The floating carrier is opaque and the integer payload channel is exact Int.
No arithmetic is performed on payloads. -/
namespace VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy
open Lean VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

inductive Storage where
  | fp16 | bf16 | fp32 | fp64 | f8e4 | f8e5 | int64
  deriving DecidableEq, Repr

def Storage.tag : Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 | .fp64 => .real
  | .f8e4 => .f8e4 | .f8e5 => .f8e5 | .int64 => .int

structure Parameters where
  (input output : RegionName)
  selection : Region .bool
  prefixSum : Region .nat
  (size width : Nat)

syntax "maskedSelectTypedSource%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(maskedSelectTypedSource%[$storage:tritonDType]) =>
    `(fun (p : Parameters) =>
      let inp_ptr := p.input
      let out_ptr := p.output
      let select_mask_ptr := p.selection
      let prefix_sum_ptr := p.prefixSum
      ComputeKernel.mk [inp_ptr, select_mask_ptr.cast, prefix_sum_ptr.cast] [out_ptr]
      (triton {
        tl.region inp_ptr = $storage, out_ptr = $storage
        pid = tl.program_id(axis=0)
        offsets = pid * $(p.width) + tl.arange(0, $(p.width))
        mask = offsets < $(p.size)
        inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
        select_mask = tl.load(select_mask_ptr + offsets,
          mask=mask, other=0.0).to(tl.int1)
        out_offset = tl.load(prefix_sum_ptr + offsets,
          mask=mask, other=0.0) - $(1)
        tl.store(out_ptr + out_offset, inp, mask=select_mask and mask)
      }).surfaceBody)

def source : Storage → Parameters → ComputeKernel
  | .fp16 => maskedSelectTypedSource%[tl.float16]
  | .bf16 => maskedSelectTypedSource%[tl.bfloat16]
  | .fp32 => maskedSelectTypedSource%[tl.float32]
  | .fp64 => maskedSelectTypedSource%[tl.float64]
  | .f8e4 => maskedSelectTypedSource%[tl.float8e4nv]
  | .f8e5 => maskedSelectTypedSource%[tl.float8e5]
  | .int64 => maskedSelectTypedSource%[tl.int64]

def inputOffset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.width]) : Nat :=
  s.pids 0 * p.width + i.1.val

def outputOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Nat :=
  (s.mem p.prefixSum.cast (inputOffset p s i)).read .nat - 1

def active {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Prop :=
  (s.mem p.selection.cast (inputOffset p s i)).read .bool = Bool.true ∧ inputOffset p s i < p.size
instance {α : Type} [Inhabited α] (p : Parameters) (s : State α) :
    DecidablePred (active p s) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

def loaded {α : Type} [Inhabited α] (d : Storage) (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Cell α :=
  .mk d.tag ((s.mem p.input (inputOffset p s i)).read d.tag)

def copied {α : Type} [Inhabited α] (d : Storage) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl (fun t i =>
    if active p s i then t.write p.output (outputOffset p s i) (loaded d p s i) else t) s

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

/-- The source floating load/store annotations erase to the exact old source;
integer data is deliberately not identified with the old mathematical Real. -/
theorem floating_source_projection (d : Storage) (h : d ≠ .int64) (p : Parameters) :
    (source d p).eraseDType.toAlgKernel.body =
      (masked_select_kernel p.input p.selection p.prefixSum p.output p.size p.width).toAlgKernel.body := by
  cases d <;> try contradiction
  all_goals simp only [source, masked_select_kernel,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  all_goals rfl

/-- Every storage profile, including actual int64 payloads, has exactly the
same program ID, offsets, bound mask, selection load and prefix load as source. -/
theorem source_control (d : Storage) (p : Parameters) :
    (source d p).surfaceBody.take 3 =
      (masked_select_kernel p.input p.selection p.prefixSum p.output p.size p.width).surfaceBody.take 3 ∧
    (source d p).surfaceBody[4]? =
      (masked_select_kernel p.input p.selection p.prefixSum p.output p.size p.width).surfaceBody[4]? ∧
    (source d p).surfaceBody[5]? =
      (masked_select_kernel p.input p.selection p.prefixSum p.output p.size p.width).surfaceBody[5]? := by
  cases d <;> exact ⟨rfl, rfl, rfl⟩


private def payloadDType : ComputeStmt → Option TileDType
  | .assign dtype _ _ _ => some dtype
  | .store dtype _ _ _ _ => some dtype
  | _ => none

/-- The int64 source load/store use the integer payload channel, rather than
renaming a mathematical real load/store as an integer implementation. -/
theorem int64_payload_dtype (p : Parameters) :
    (source .int64 p).surfaceBody[3]?.bind payloadDType = some .int ∧
      (source .int64 p).surfaceBody[6]?.bind payloadDType = some .int := ⟨rfl, rfl⟩

theorem execution_memory {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) :
    (execution rest d p s).val.mem = (copied d p s).mem := by
  cases d
  all_goals simp +contextual [execution, copied, masked_scatter_memory, loaded,
    Storage.tag, active, inputOffset, outputOffset, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, and_assoc, and_left_comm, and_comm]

/-- Address collisions are excluded only among selected in-bounds lanes.
Unselected prefix entries may repeat freely. -/
theorem readback {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (hi : active p s i)
    (hinj : ∀ a b, active p s a → active p s b → outputOffset p s a = outputOffset p s b → a = b) :
    (execution rest d p s).val.mem p.output (outputOffset p s i) =
      .mk d.tag ((s.mem p.input (inputOffset p s i)).read d.tag) := by
  rw [execution_memory]
  exact masked_readback_active _ _ _ _ _ i hi hinj

theorem typed_readback {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (hi : active p s i)
    (hinj : ∀ a b, active p s a → active p s b → outputOffset p s a = outputOffset p s b → a = b)
    (value : Value α d.tag) (htyped : s.mem p.input (inputOffset p s i) = .mk d.tag value) :
    (execution rest d p s).val.mem p.output (outputOffset p s i) = .mk d.tag value := by
  rw [readback rest d p s i hi hinj, htyped, Cell.read_same]


/-- Every int64 bit pattern survives selection and re-encoding, including the
signed endpoints. The source executes through its actual Int payload channel. -/
theorem int64_bits_readback {α : Type} [Inhabited α] (rest : Algebra α)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (hi : active p s i)
    (hinj : ∀ a b, active p s a → active p s b → outputOffset p s a = outputOffset p s b → a = b)
    (bits : BitVec 64)
    (htyped : s.mem p.input (inputOffset p s i) = .mk .int bits.toInt) :
    BitVec.ofInt 64 (((execution rest .int64 p s).val.mem p.output (outputOffset p s i)).read .int) = bits := by
  rw [typed_readback rest .int64 p s i hi hinj bits.toInt htyped]
  simp [Storage.tag]

theorem frame {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat)
    (hmiss : r ≠ p.output ∨ ∀ i : TileIndex [p.width], active p s i → o ≠ outputOffset p s i) :
    (execution rest d p s).val.mem r o = s.mem r o := by
  rw [execution_memory]
  apply State.masked_scatter_frame
  exact hmiss.imp id (fun h i _ => h i)

theorem correctness {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α)
    (hinj : ∀ a b, active p s a → active p s b → outputOffset p s a = outputOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (copyBackend rest) (source d p) s = some t ∧
      (∀ i : TileIndex [p.width], active p s i → t.mem p.output (outputOffset p s i) =
        .mk d.tag ((s.mem p.input (inputOffset p s i)).read d.tag)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ i : TileIndex [p.width], active p s i → o ≠ outputOffset p s i) →
        t.mem r o = s.mem r o) :=
  ⟨(execution rest d p s).val, (execution rest d p s).property,
    fun i hi => readback rest d p s i hi hinj, frame rest d p s⟩

#axiomsClean floating_source_projection
#axiomsClean source_control
#axiomsClean int64_payload_dtype
#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean readback
#axiomsClean typed_readback
#axiomsClean int64_bits_readback
#axiomsClean frame
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy
