import bench.tritonbench_g.mixed_sparse_attention.Proofs.FPSuffix
import bench.tritonbench_g.mixed_sparse_attention.Proofs.FPLoops
namespace VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

def blockStatement (c : Config) : ComputeStmt :=
  .forRangeDyn "block_index" (.constNat 0) (.ref .nat [] "max_num_blks") (.constNat 1) (blockBody c)

def columnStatement (c : Config) : ComputeStmt :=
  .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "max_num_cols") (.constNat c.BN) (columnBody c)

def between (c : Config) : List ComputeStmt := ((activeBody c).drop 22).take 1

theorem source_split (c : Config) :
    (parsedKernel c).surfaceBody.take 3 ++ activeBody c =
      prelude c ++ [blockStatement c] ++ between c ++ [columnStatement c] ++ suffix c := by
  simp [prelude,blockStatement,columnStatement,between,suffix,blockBody,columnBody,
    activeBody,parsedActiveBody,parsedKernel,source,ComputeKernel.surfaceBody]

theorem between_run {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Config) (s t : State α) (a : Accumulator α c.BM c.BD) (h : Ready M c s t a) :
    ∃ final, run (backend M D) (between c) t = some final ∧ Ready M c s final a ∧
      final.regs .nat [] "max_num_cols" = some (fun _ => 16) := by
  simp [between,activeBody,parsedActiveBody,parsedKernel,source,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind]
  constructor <;> simp [State.setReg,h.mem,h.pids,h.undef,h.seqlen,h.rows,h.lanes,h.keyptr,h.valueptr,
    h.outptr,h.numblocks,h.blockptr,h.numcolumns,h.columnptr,h.query,h.mask,h.maximum,h.normalizer,h.output,h.maxblocks]

theorem block_statement_run {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Config) (s t : State α) (a : Accumulator α c.BM c.BD) (h : Ready M c s t a) (hBN : 0<c.BN) :
    ∃ final, step (backend M D) (blockStatement c) t = some final ∧
      Ready M c s final (foldRange (update M D c s Bool.false) 0 8 1 a) := by
  simpa [blockStatement,step,evalOp_unfold,h.maxblocks,Option.bind,loopBody,indexName] using
    loop_range_run M D c s t Bool.false 0 8 1 a h hBN

theorem column_statement_run {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Config) (s t : State α) (a : Accumulator α c.BM c.BD) (h : Ready M c s t a)
    (hc : t.regs .nat [] "max_num_cols" = some (fun _ => 16)) (hBN : 0<c.BN) :
    ∃ final, step (backend M D) (columnStatement c) t = some final ∧
      Ready M c s final (foldRange (update M D c s Bool.true) 0 16 c.BN a) := by
  simpa [columnStatement,step,evalOp_unfold,hc,Option.bind,loopBody,indexName] using
    loop_range_run M D c s t Bool.true 0 16 c.BN a h hBN

theorem active_kernel_run {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Config) (s : State α) (hactive : rowStart c s < sequenceLength c s) :
    FP.Structural.exec (backend M D) (kernel c) s =
      run (backend M D) ((parsedKernel c).surfaceBody.take 3 ++ activeBody c) s := by
  -- Name the source's existing branch body before execution simplification.
  -- This is a definitional decomposition; no statement is replaced.
  change FP.Structural.exec (backend M D)
    (match parsedKernel c with
     | .mk inputs outputs statements => .mk inputs outputs
       (statements.mapIdx (fun i st => match i,st with
         | 3,.ifThen cond _ => .ifThen cond (activeBody c)
         | _,_ => st))) s = _
  simp only [rowStart,sequenceLength,Region.cast] at hactive
  simp [kernel,parsedKernel,source,ComputeKernel.surfaceBody,FP.Structural.exec,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,natLe,numericLt,numericLe,integralFloorDiv,integralMod,
    Region.cast,hactive]
  split <;> simp_all

/-- Complete actual fixed-block and column loops followed by native-width
writeback. The value specification is a pure initial-memory recurrence, not
execution of another kernel or an assumed final output. -/
specification whole_fp16 {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Config) (s : State α) (hBN : 0<c.BN) (hactive : rowStart c s < sequenceLength c s) :
    ∃ final, FP.Structural.exec (backend M D) (kernel c) s = some final ∧
      final.mem = (outputMemory M c s (finalValues M D c s)).mem := by
  obtain ⟨a,ha,har⟩ := prefix_run M D c s
  obtain ⟨b,hb,hbr⟩ := block_statement_run M D c s a _ har hBN
  obtain ⟨d,hd,hdr,hdc⟩ := between_run M D c s b _ hbr
  obtain ⟨e,he,her⟩ := column_statement_run M D c s d _ hdr hdc hBN
  obtain ⟨final,hf,hm⟩ := suffix_run M D c s e _ her
  refine ⟨final,?_,hm⟩
  rw [active_kernel_run M D c s hactive,source_split]
  simp only [run_append,ha,hb,hd,he,hf,run,bind,pure,Option.bind]

def acceptedValues {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) (c : Config) (s : State α) :=
  update M D c s Bool.true 0
    (foldRange (update M D c s Bool.false) 0 8 1 (initialAccumulator M c))

theorem one_column_iteration {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Config) (s : State α) (hBN : 16≤c.BN) :
    finalValues M D c s = acceptedValues M D c s := by
  have hzero : c.BN ≠ 0 := by omega
  have hstop : ¬ c.BN < 16 := by omega
  simp [finalValues,acceptedValues,foldRange,hzero,hstop]

#axiomsClean one_column_iteration
#axiomsClean source_split
#axiomsClean between_run
#axiomsClean block_statement_run
#axiomsClean column_statement_run
#axiomsClean active_kernel_run
#axiomsClean whole_fp16
end
end VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
