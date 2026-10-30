import bench.repairs.tritonbench_g.kcache_copy_triton.Kernels
import VeriTile.Triton.Semantics
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Semantics.Integer
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.KcacheCopyTriton.KernelReal
open VeriTile.Triton Source
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

private theorem positive_sub_negative (z : Int) :
    z.toNat - (-z).toNat = z.toNat := by omega
attribute [local simp] positive_sub_negative

def sequence (p : Parameters) (ids : Nat → Nat) : Int := (ids 0 : Int).tdiv p.n
def position (p : Parameters) (s : BlockState) : Int :=
  s.readMemValue .int p.lengths.cast (sequence p s.pids).toNat +
    ((s.pids 0 : Int) - (p.n : Int) * (sequence p s.pids + 1))
def selectedBlock (p : Parameters) (s : BlockState) : Int :=
  s.readMemValue .int p.blockTables.cast
    ((sequence p s.pids * p.strideBts).toNat + ((position p s).tdiv p.blockSize * p.strideBtb).toNat - (-((position p s).tdiv p.blockSize * p.strideBtb)).toNat)
def sourceOffset (p : Parameters) (ids : Nat → Nat) (i : TileIndex [p.width]) : Nat :=
  ids 0 * p.strideKt + ids 1 * p.strideKh + (ids 2 * p.width + i.1.val) * p.strideKd
def targetOffset (p : Parameters) (s : BlockState) (i : TileIndex [p.width]) : Nat :=
  (selectedBlock p s * p.strideKcb + (s.pids 1 : Int) * p.strideKch +
    (s.pids 2 : Int) * p.strideKcsplit + (position p s).tmod p.blockSize * p.strideKcs + i.1.val).toNat

def copied (p : Parameters) (s : BlockState) : BlockState :=
  (TileShape.allIndices [p.width]).foldl (fun t i =>
    t.writeMem p.KCache (targetOffset p s i) (s.readMem p.K (sourceOffset p s.pids i))) s

def headState (p : Parameters) (s : BlockState) : BlockState :=
  let z := s.setReg "cur_token_idx" .int [] (Tile.scalar (s.pids 0))
  let z := z.setReg "cur_seq_idx" .int [] (Tile.scalar (sequence p s.pids))
  let z := z.setReg "cur_token_shift" .int [] (Tile.scalar ((s.pids 0:Int)-p.n*(sequence p s.pids+1)))
  let z := z.setReg "cur_kv_head_idx" .int [] (Tile.scalar (s.pids 1))
  z.setReg "split_x_idx" .int [] (Tile.scalar (s.pids 2))

def metadataState (p : Parameters) (s : BlockState) : BlockState :=
  let z := headState p s
  let z := z.setReg "past_kv_seq_len" .int [] (Tile.scalar (position p s))
  let z := z.setReg "last_bt_block_idx" .int [] (Tile.scalar ((position p s).tdiv p.blockSize))
  let z := z.setReg "block_table_ptr" .ptr [] (Tile.scalar (p.blockTables.cast,(sequence p s.pids*p.strideBts).toNat))
  let z := z.setReg "block_id" .int [] (Tile.scalar (selectedBlock p s))
  let z := z.setReg "offset_last_block" .int [] (Tile.scalar ((position p s).tmod p.blockSize))
  z.setReg "offsets_dmodel" .int [p.width] ⟨fun i => (s.pids 2:Int)*p.width+i.1.val⟩

def preparedState (p : Parameters) (s : BlockState) : BlockState :=
  let z := metadataState p s
  let z := z.setReg "offsets_k" .int [p.width] ⟨fun i => sourceOffset p s.pids i⟩
  let z := z.setReg "k" .real [p.width] ⟨fun i => some (s.readMem p.K (sourceOffset p s.pids i))⟩
  z.setReg "offsets_kcache" .int [p.width] ⟨fun i =>
    selectedBlock p s*p.strideKcb+(s.pids 1:Int)*p.strideKch+
      (s.pids 2:Int)*p.strideKcsplit+(position p s).tmod p.blockSize*p.strideKcs+i.1.val⟩

def finalState (p : Parameters) (s : BlockState) : BlockState :=
  (TileShape.allIndices [p.width]).foldl (fun t i =>
    t.writeMem p.KCache (targetOffset p s i) (s.readMem p.K (sourceOffset p s.pids i))) (preparedState p s)

def source (p : Parameters) := (originalKernel p).toAlgKernel
/-- The selected typed source projects successfully; no rejected-source fallback
is used. FP32 compute loads project directly to the Real channel. -/
theorem projection (p : Parameters) :
    (originalKernel p).toAlgorithm? = .ok (source p) := by
  simp [source,originalKernel,kernel,quotedOriginal,quotedCommuted,ComputeKernel.surfaceBody,ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind]
  all_goals rfl

def headBody (p : Parameters) := (source p).body.take 5
def metadataBody (p : Parameters) := ((source p).body.drop 5).take 6
def tailBody (p : Parameters) := (source p).body.drop 11

theorem decomposition (p : Parameters) :
    (source p).body = headBody p ++ metadataBody p ++ tailBody p := by
  unfold headBody metadataBody tailBody
  generalize (source p).body = code
  have inner := congrArg (fun rest => code.take 5 ++ rest) (List.take_append_drop 6 (code.drop 5))
  simpa only [List.drop_drop,List.append_assoc,Nat.reduceAdd] using
    (List.take_append_drop 5 code).symm.trans inner.symm

theorem head_run (p : Parameters) (s : BlockState) :
    stepStmts (headBody p) s=some (headState p s) := by
  simp [headBody,headState,sequence,source,originalKernel,kernel,quotedOriginal,quotedCommuted,ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
    Region.cast,Except.bind,bind,
    stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
    Tile.bop,Tile.ptrAdd,Tile.ptrSub,Tile.uop,Tile.cop,Tile.select,IntegerExpr.same_left,IntegerExpr.same_right,NumericDType.add,NumericDType.mul,NumericDType.sub,
    IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,
    VeriTile.Triton.eval_truncDiv,VeriTile.Triton.eval_truncMod,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try simp [show ¬(s.pids 0:Int)<0 from by omega]
  all_goals rfl

theorem metadata_run (p : Parameters) (s : BlockState) :
    stepStmts (metadataBody p) (headState p s)=some (metadataState p s) := by
  simp [metadataBody,metadataState,headState,selectedBlock,position,sequence,source,originalKernel,kernel,quotedOriginal,quotedCommuted,ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
    Region.cast,Except.bind,bind,
    stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
    Tile.bop,Tile.ptrAdd,Tile.ptrSub,Tile.uop,Tile.cop,Tile.select,IntegerExpr.same_left,IntegerExpr.same_right,NumericDType.add,NumericDType.mul,NumericDType.sub,
    IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,
    VeriTile.Triton.eval_truncDiv,VeriTile.Triton.eval_truncMod,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals simp [Tile.bop,Tile.uop,Tile.cop,Tile.select,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt,Broadcast.leftIndex,Broadcast.rightIndex,←IntegerExpr.tdiv_formula,Int.tmod_def]

theorem tail_run (p : Parameters) (s : BlockState) :
    stepStmts (tailBody p) (metadataState p s)=some (finalState p s) := by
  simp [tailBody,finalState,preparedState,metadataState,headState,sourceOffset,targetOffset,
    Int.add_comm,Int.add_left_comm,Int.add_assoc,BlockState.readMemValue,source,originalKernel,kernel,quotedOriginal,quotedCommuted,ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
    Region.cast,Except.bind,bind,
    stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
    Tile.bop,Tile.ptrAdd,Tile.ptrSub,Tile.uop,Tile.cop,Tile.select,IntegerExpr.same_left,IntegerExpr.same_right,NumericDType.add,NumericDType.mul,NumericDType.sub,
    IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,
    VeriTile.Triton.eval_truncDiv,VeriTile.Triton.eval_truncMod,
    Broadcast.leftIndex,Broadcast.rightIndex]
  simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.toNat_neg_natCast,Nat.sub_zero]
  all_goals simp [Tile.bop,Tile.uop,NumericDType.add,NumericDType.mul,NumericDType.sub,Int.add_comm,Int.add_left_comm,Int.add_assoc,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]

  congr 20
  all_goals funext
  all_goals first | ring | (congr 2 <;> ring)

theorem execute (p : Parameters) (s : BlockState) :
    exec (originalKernel p).toAlgKernel s = some (finalState p s) := by
  change exec (source p) s = _
  simp only [exec,decomposition,ProgramRewrite.append,head_run,Option.bind_some,metadata_run,tail_run]

def execution (p : Parameters) (s : BlockState) :
    {t : BlockState // exec (originalKernel p).toAlgKernel s = some t} :=
  ⟨finalState p s,execute p s⟩

theorem scatter_memory {I : Type} (xs : List I) (s : BlockState)
    (r : RegionName) (offset : I → Nat) (value : I → ℝ) :
    (xs.foldl (fun t i => t.writeMem r (offset i) (value i)) s).mem =
      xs.foldl (fun m i => fun r' o => if r'=r ∧ o=offset i then MemCell.real (value i) else m r' o) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => exact ih (s.writeMem r (offset i) (value i))

theorem execution_memory (p : Parameters) (s : BlockState) :
    (execution p s).val.mem = (copied p s).mem := by
  simp only [execution,finalState,copied,scatter_memory]
  rfl

theorem readback (p : Parameters) (s : BlockState)
    (injective : Function.Injective (targetOffset p s)) (i : TileIndex [p.width]) :
    (execution p s).val.readMem p.KCache (targetOffset p s i) =
      s.readMem p.K (sourceOffset p s.pids i) := by
  have hm := execution_memory p s
  change BlockState.readMem _ _ _ = _
  simp only [BlockState.readMem,hm]
  exact BlockState.scatter_readback_nd s _ _ injective i

theorem frame (p : Parameters) (s : BlockState) (r : RegionName) (o : Nat)
    (outside : r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], o ≠ targetOffset p s i) :
    (execution p s).val.mem r o = s.mem r o := by
  rw [execution_memory]
  rcases outside with other | miss
  · exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r other o s
  · by_cases same : r=p.KCache
    · subst r
      exact BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o (fun i _ => (miss i).symm) s
    · exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r same o s

#axiomsClean projection
#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean readback
#axiomsClean frame
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.KernelReal
