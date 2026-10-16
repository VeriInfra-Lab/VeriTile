import bench.tritonbench_g.rotary_transform.RotaryTransform
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import VeriTile.Triton.Float.ScalarBackend

/-! Actual fp32 source for the already selected paired row and fixed/varlen
metadata row. All four source register casts are retained. The source profile
uses separate rounded multiply and add/subtract operations; CUDA contraction,
FTZ and machine-address refinement are not assumed by this software model. -/
namespace VeriTile.Bench.TritonBenchG.RotaryTransform.FPRow
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (masked_scatter_regs masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

inductive Kind where
  | row | fixed | varlen
  deriving DecidableEq, Repr

structure Parameters where
  (output input cosine sine : RegionName)
  cuLengths : Region .nat
  (headIndex sequenceOffset sequenceLength halfDimension cosineLength
    strideOutBatch strideOutSequence strideOutHead strideOutDimension
    strideInBatch strideInSequence strideInHead strideInDimension blockRows width : Nat)

def rowTyped
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  tl.region OUT = tl.float32, X = tl.float32, COS = tl.float32, SIN = tl.float32
  pid_m = tl.program_id(axis=0)
  pid_batch = tl.program_id(axis=1)
  pid_head = tl.program_id(axis=2)
  rm = pid_m * $(BLOCK_M)
  rm_cs = rm + $(SEQLEN_OFFSETS)
  rk_half = tl.arange(0, $(BLOCK_HALF))
  x_base = X + pid_batch * $(stride_x_batch) + pid_head * $(stride_x_nheads)
  out_base = OUT + pid_batch * $(stride_out_batch) + pid_head * $(stride_out_nheads)
  cos = tl.load(COS + rm_cs * $(rotary_dim_half) + rk_half,
    mask=(rm_cs < $(seqlen_ro)) and (rk_half < $(rotary_dim_half)), other=1.0).to(tl.float32)
  sin = tl.load(SIN + rm_cs * $(rotary_dim_half) + rk_half,
    mask=(rm_cs < $(seqlen_ro)) and (rk_half < $(rotary_dim_half)), other=0.0).to(tl.float32)
  x0 = tl.load(x_base + rm * $(stride_x_seqlen) + rk_half * $(stride_x_headdim),
    mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)), other=0.0).to(tl.float32)
  x1 = tl.load(x_base + rm * $(stride_x_seqlen) +
      (rk_half + $(rotary_dim_half)) * $(stride_x_headdim),
    mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)), other=0.0).to(tl.float32)
  o0 = x0 * cos - x1 * sin
  o1 = x0 * sin + x1 * cos
  tl.store(out_base + rm * $(stride_out_seqlen) + rk_half * $(stride_out_headdim),
    o0, mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)))
  tl.store(out_base + rm * $(stride_out_seqlen) +
      (rk_half + $(rotary_dim_half)) * $(stride_out_headdim),
    o1, mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)))
}

def metaTyped
    (OUT X COS SIN : RegionName) (CU_SEQLENS : Region .nat)
    (HEAD_IDX SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) (IS_VARLEN : Bool) :
    ComputeKernel := triton {
  tl.region OUT = tl.float32, X = tl.float32, COS = tl.float32, SIN = tl.float32
  pid_m = tl.program_id(0)
  pid_batch = tl.program_id(1)
  rm = pid_m * $(BLOCK_M)
  rm_cs = rm + $(SEQLEN_OFFSETS)
  rk_half = tl.arange(0, $(BLOCK_HALF))
  x_base = X + pid_batch * $(stride_x_batch) + $(HEAD_IDX) * $(stride_x_nheads)
  out_base = OUT + pid_batch * $(stride_out_batch) + $(HEAD_IDX) * $(stride_out_nheads)
  seqlen_v = $(seqlen)
  if IS_VARLEN {
    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen_hi = tl.load(CU_SEQLENS + pid_batch + $(1))
    x_base = X + start_idx * $(stride_x_seqlen) + $(HEAD_IDX) * $(stride_x_nheads)
    out_base = OUT + start_idx * $(stride_out_seqlen) + $(HEAD_IDX) * $(stride_out_nheads)
    seqlen_v = seqlen_hi - start_idx
  }
  cos = tl.load(COS + rm_cs * $(rotary_dim_half) + rk_half,
    mask=(rm_cs < $(seqlen_ro)) and (rk_half < $(rotary_dim_half)), other=1.0).to(tl.float32)
  sin = tl.load(SIN + rm_cs * $(rotary_dim_half) + rk_half,
    mask=(rm_cs < $(seqlen_ro)) and (rk_half < $(rotary_dim_half)), other=0.0).to(tl.float32)
  x0 = tl.load(x_base + rm * $(stride_x_seqlen) + rk_half * $(stride_x_headdim),
    mask=(rm < seqlen_v) and (rk_half < $(rotary_dim_half)), other=0.0).to(tl.float32)
  x1 = tl.load(x_base + rm * $(stride_x_seqlen) +
      (rk_half + $(rotary_dim_half)) * $(stride_x_headdim),
    mask=(rm < seqlen_v) and (rk_half < $(rotary_dim_half)), other=0.0).to(tl.float32)
  o0 = x0 * cos - x1 * sin
  o1 = x0 * sin + x1 * cos
  tl.store(out_base + rm * $(stride_out_seqlen) + rk_half * $(stride_out_headdim),
    o0, mask=(rm < seqlen_v) and (rk_half < $(rotary_dim_half)))
  tl.store(out_base + rm * $(stride_out_seqlen) +
      (rk_half + $(rotary_dim_half)) * $(stride_out_headdim),
    o1, mask=(rm < seqlen_v) and (rk_half < $(rotary_dim_half)))
}

def source (k : Kind) (p : Parameters) : ComputeKernel :=
  match k with
  | .row => rowTyped p.output p.input p.cosine p.sine
      p.sequenceOffset p.sequenceLength p.halfDimension p.cosineLength
      p.strideOutBatch p.strideOutSequence p.strideOutHead p.strideOutDimension
      p.strideInBatch p.strideInSequence p.strideInHead p.strideInDimension p.blockRows p.width
  | .fixed | .varlen => metaTyped p.output p.input p.cosine p.sine p.cuLengths
      p.headIndex p.sequenceOffset p.sequenceLength p.halfDimension p.cosineLength
      p.strideOutBatch p.strideOutSequence p.strideOutHead p.strideOutDimension
      p.strideInBatch p.strideInSequence p.strideInHead p.strideInDimension p.blockRows p.width (k == .varlen)

def legacySource (k : Kind) (p : Parameters) : ComputeKernel :=
  match k with
  | .row => rotary_kernel_o0o1_row p.output p.input p.cosine p.sine
      p.sequenceOffset p.sequenceLength p.halfDimension p.cosineLength
      p.strideOutBatch p.strideOutSequence p.strideOutHead p.strideOutDimension
      p.strideInBatch p.strideInSequence p.strideInHead p.strideInDimension p.blockRows p.width
  | .fixed | .varlen => rotary_meta_row p.output p.input p.cosine p.sine p.cuLengths
      p.headIndex p.sequenceOffset p.sequenceLength p.halfDimension p.cosineLength
      p.strideOutBatch p.strideOutSequence p.strideOutHead p.strideOutDimension
      p.strideInBatch p.strideInSequence p.strideInHead p.strideInDimension p.blockRows p.width (k == .varlen)

def rowIndex {α : Type} (p : Parameters) (s : State α) : Nat := s.pids 0 * p.blockRows

def headIndex {α : Type} (k : Kind) (p : Parameters) (s : State α) : Nat :=
  if k = .row then s.pids 2 else p.headIndex

def startIndex {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem p.cuLengths.cast (s.pids 1)).read .nat

def sequenceLength {α : Type} [Inhabited α] (k : Kind) (p : Parameters) (s : State α) : Nat :=
  if k = .varlen then (s.mem p.cuLengths.cast (s.pids 1 + 1)).read .nat - startIndex p s
  else p.sequenceLength

def inputBase {α : Type} [Inhabited α] (k : Kind) (p : Parameters) (s : State α) : Nat :=
  (if k = .varlen then startIndex p s * p.strideInSequence else s.pids 1 * p.strideInBatch) +
    headIndex k p s * p.strideInHead

def outputBase {α : Type} [Inhabited α] (k : Kind) (p : Parameters) (s : State α) : Nat :=
  (if k = .varlen then startIndex p s * p.strideOutSequence else s.pids 1 * p.strideOutBatch) +
    headIndex k p s * p.strideOutHead

def inputOffset {α : Type} [Inhabited α] (k : Kind) (p : Parameters) (s : State α)
    (second : Bool) (i : TileIndex [p.width]) : Nat :=
  inputBase k p s + rowIndex p s * p.strideInSequence +
    (i.1.val + if second then p.halfDimension else 0) * p.strideInDimension

def outputOffset {α : Type} [Inhabited α] (k : Kind) (p : Parameters) (s : State α)
    (second : Bool) (i : TileIndex [p.width]) : Nat :=
  outputBase k p s + rowIndex p s * p.strideOutSequence +
    (i.1.val + if second then p.halfDimension else 0) * p.strideOutDimension

def cacheOffset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.width]) : Nat :=
  (rowIndex p s + p.sequenceOffset) * p.halfDimension + i.1.val

def active {α : Type} [Inhabited α] (k : Kind) (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Prop :=
  rowIndex p s < sequenceLength k p s ∧ i.1.val < p.halfDimension
instance {α : Type} [Inhabited α] (k : Kind) (p : Parameters) (s : State α) :
    DecidablePred (active k p s) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

def cacheActive {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.width]) : Prop :=
  rowIndex p s + p.sequenceOffset < p.cosineLength ∧ i.1.val < p.halfDimension
instance {α : Type} (p : Parameters) (s : State α) :
    DecidablePred (cacheActive p s) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

def maskedCast {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (r : RegionName) (o : Nat) (active : Prop) [Decidable active] (other : ℝ) : α :=
  M.cast (some .fp32) .real .real
    (if active then (s.mem r o).read .real else M.literal (some .fp32) .real other)

/-- Independent scalar operation tree, with the source order kept exactly. -/
def value {α : Type} [Inhabited α] (M : Algebra α) (k : Kind) (p : Parameters) (s : State α)
    (second : Bool) (i : TileIndex [p.width]) : α :=
  let x0 := maskedCast M s p.input (inputOffset k p s Bool.false i) (active k p s i) 0
  let x1 := maskedCast M s p.input (inputOffset k p s Bool.true i) (active k p s i) 0
  let c := maskedCast M s p.cosine (cacheOffset p s i) (cacheActive p s i) 1
  let z := maskedCast M s p.sine (cacheOffset p s i) (cacheActive p s i) 0
  if second then M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul x0 z) (M.binary (some .fp32) .real .mul x1 c)
  else M.binary (some .fp32) .real .sub
      (M.binary (some .fp32) .real .mul x0 c) (M.binary (some .fp32) .real .mul x1 z)

def halfScatter {α : Type} [Inhabited α] (M : Algebra α) (k : Kind) (p : Parameters)
    (s : State α) (second : Bool) (initial : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl (fun t i =>
    if active k p s i then t.write p.output (outputOffset k p s second i)
      (.mk .real (value M k p s second i)) else t) initial

def copied {α : Type} [Inhabited α] (M : Algebra α) (k : Kind) (p : Parameters) (s : State α) : State α :=
  halfScatter M k p s Bool.true (halfScatter M k p s Bool.false s)

def execution {α : Type} [Inhabited α] (M : Algebra α) (k : Kind)
    (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source k p) s = some t} := by
  cases k <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [source, rowTyped, metaTyped, FP.Structural.exec, ComputeKernel.surfaceBody,
       FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
       FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
       FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
       FP.Structural.bop, FP.Structural.natLt, ComputeDType.eraseDType,
       FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind]
     try rfl)

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (k : Kind)
    (p : Parameters) (s : State α) :
    (execution M k p s).val.mem = (copied M k p s).mem := by
  cases k
  all_goals simp +contextual [execution, copied, halfScatter, masked_scatter_memory,
    value, maskedCast, active, cacheActive, inputOffset, outputOffset, inputBase, outputBase,
    cacheOffset, rowIndex, headIndex, startIndex, sequenceLength,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    and_assoc, and_left_comm, and_comm]
  all_goals norm_num

#axiomsClean execution
#axiomsClean execution_memory
end
end VeriTile.Bench.TritonBenchG.RotaryTransform.FPRow
