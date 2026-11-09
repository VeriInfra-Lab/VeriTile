/- Reuses the historical complete retention execution proof, adapted only
to this repair's explicit zero-padded source and task-local namespace. -/
import bench.tritonbench_g.parallel_retention_attention.Proofs.BackwardPrefix
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.Sources.Mathematical_FPStages
namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive withDot BackwardConfig BackwardAccumulator matrixTranspose descendingCount descendingStart foldRange backwardOutputMemory)
open VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages (scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/- Local copy of the only extra register-frame lemma formerly obtained via
the unrelated complete parallel-attention forward theorem. -/
@[simp] theorem masked_scatter_regs {α ι : Type} (R : RegionName) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (value : ι → Cell α) (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write R (off i) (value i) else t) s).regs = s.regs := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons, ih]
    split <;> rfl

def backwardMiddle (c : BackwardConfig) := ((backwardKernel c).surfaceBody.drop 44).take 7
def backwardSuffix (c : BackwardConfig) := (backwardKernel c).surfaceBody.drop 52
def backwardFirstRange (c : BackwardConfig) : ComputeStmt :=
  ((backwardKernel c).surfaceBody[43]?).getD (.ifThen (.constBool Bool.false) [])
def backwardSecondRange (c : BackwardConfig) : ComputeStmt :=
  ((backwardKernel c).surfaceBody[51]?).getD (.ifThen (.constBool Bool.false) [])
theorem backward_middle_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) :
    ∃ final, run (withDot M D) (backwardMiddle c) t = some final ∧
      BackwardDiagonalReady M c s final (c.i_c*c.BTL) (backwardBetweenValues M c acc) := by
  simp [backwardMiddle, backwardKernel, backwardSource, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, Option.bind, ComputeDType.eraseDType,
    h.keyGrad, h.valueGrad, h.decayHead,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,numeric,bop,withDot, ofFloat, toFloat]
  constructor
  · constructor <;> simp [h.mem, h.pids, h.undef, h.key, h.value, h.keyGrad, h.valueGrad, h.decayLog, h.decayBlock, h.decayHead, h.decayQuery, h.keyDecayed,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13, backwardBetweenValues, rowDecay, bop]
    all_goals rfl
  · rfl
  · simp [State.setReg]
  · simp [State.setReg]

theorem backward_suffix_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) (bounds : HelperIndexBounds c) :
    ∃ final, run (withDot M D) (backwardSuffix c) t = some final ∧
      final.mem = (backwardOutputMemory M c s acc).mem := by
  simp [backwardSuffix, backwardKernel, backwardSource, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, boundaryZero, evalOffsets, Option.bind,
    ComputeDType.eraseDType, numeric, bop, Region.cast, withDot, store,
    h.keyGrad,h.valueGrad,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,bounds.query_cast,bounds.key_cast,bounds.value_cast,Index32.natural_product,TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, ofFloat, toFloat, backwardOutputMemory]
  apply scatter_memory_congr
  apply scatter_memory_congr
  exact h.mem

theorem backward_source_split (c : BackwardConfig) :
    (backwardKernel c).surfaceBody = backwardPrefix c ++ [backwardFirstRange c] ++
      backwardMiddle c ++ [backwardSecondRange c] ++ backwardSuffix c := by rfl

theorem backward_first_range_statement {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) (positive : 0<c.BTS) :
    step (withDot M D) (backwardFirstRange c) t =
      rangeInt (withDot M D) "__rev_i"
        (DescendingRange.lower c)
        (DescendingRange.upper c)
        c.BTS (backwardFirstBody c) t := by
  have offset : ((c.T+c.BTS-1 : Nat) : Int)=(c.T : Int)+(c.BTS : Int)-1 := by omega
  have nonnegative : ¬((c.T : Int)+(c.BTS : Int)-1<0) := by omega
  simp [backwardFirstRange,backwardFirstBody,backwardKernel,backwardSource,ComputeKernel.surfaceBody,
    step,evalOp_unfold,numeric,bop,Option.bind,h.argument13,h.argument1,
    integralFloorDiv,numericLt,natLt,nonnegative,
    DescendingRange.lower,
    DescendingRange.upper,
    Int.natCast_mul,Int.natCast_add,Int.natCast_ediv,offset]
  simp only [←Int.add_sub_assoc,nonnegative,if_false]
  congr 1 <;> omega

theorem backward_second_range_statement {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) :
    step (withDot M D) (backwardSecondRange c) t =
      FP.Structural.rangeInt (withDot M D) "i" ((c.i_c*c.BTL : Nat) : Int) (((c.i_c+1)*c.BTL : Nat) : Int)
        c.BTS (backwardSecondBody c) t := by
  simp [backwardSecondRange,backwardSecondBody,backwardKernel,backwardSource,ComputeKernel.surfaceBody,
    step,evalOp_unfold,numeric,bop,Option.bind,h.argument1,Int.natCast_mul,Int.natCast_add]

/-- Complete original accepted DK/DV stage, including both loop ranges and
all loads, casts, dot options, branch orders and output memory framing. -/
specification backward_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s : State α) (bounds : BackwardIndexBounds c) (positive : 0<c.BTS) :
    ∃ final, FP.Structural.exec (withDot M D) (backwardKernel c) s = some final ∧
      final.mem = (backwardOutputMemory M c s (backwardValues M D c s)).mem := by
  obtain ⟨initial, hp, hi⟩ := backward_prefix_run M D c s bounds.toHelperIndexBounds
  obtain ⟨first, hr, hf⟩ := backward_first_range_run M D c s initial 0 (by omega) _ hi bounds
  simp only [Nat.cast_zero,zero_mul,add_zero] at hr
  have stopBound : (c.i_c+1)*c.BTL≤2147483648 := by
    have := bounds.time
    nlinarith
  obtain ⟨diagonal, hm, hd⟩ := backward_middle_run M D c s first _ hf
  obtain ⟨second, hs, hz⟩ := backward_second_range_run M D c s diagonal (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) _ hd bounds.toHelperIndexBounds stopBound
  obtain ⟨final, he, hout⟩ := backward_suffix_run M D c s second _ hz bounds.toHelperIndexBounds
  refine ⟨final, ?_, ?_⟩
  · have hexec : FP.Structural.exec (withDot M D) (backwardKernel c) s =
        run (withDot M D) (backwardKernel c).surfaceBody s := by cases backwardKernel c; rfl
    rw [hexec, backward_source_split]
    simp only [run_append, hp, bind, pure, Option.bind,run,backward_first_range_statement M D c s initial _ hi positive,
      hr,hm,backward_second_range_statement M D c s diagonal _ hd.toBackwardReady,hs,he]
  · simpa only [backwardValues, backwardInitial] using hout

#axiomsClean backward_fp32
#axiomsClean backward_source_split
#axiomsClean backward_prefix_run
#axiomsClean backward_middle_run
#axiomsClean backward_suffix_run
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
