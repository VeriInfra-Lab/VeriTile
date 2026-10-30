import bench.tritonbench_g.kcache_copy_triton.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.Integer
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.KcacheCopyTriton.KernelFP
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def sequence (p : Parameters) (ids : Nat → Nat) : Int := (ids 0 : Int).tdiv p.n
def position {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Int :=
  (s.mem p.lengths.cast (sequence p s.pids).toNat).read .int +
    ((s.pids 0 : Int) - (p.n : Int) * (sequence p s.pids + 1))
def selectedBlock {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Int :=
  (s.mem p.blockTables.cast
    ((sequence p s.pids * p.strideBts).toNat + ((position p s).tdiv p.blockSize * p.strideBtb).toNat)).read .int
def sourceOffset (p : Parameters) (ids : Nat → Nat) (i : TileIndex [p.width]) : Nat :=
  ids 0 * p.strideKt + ids 1 * p.strideKh + (ids 2 * p.width + i.1.val) * p.strideKd
def targetOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Nat :=
  (selectedBlock p s * p.strideKcb + (s.pids 1 : Int) * p.strideKch +
    (s.pids 2 : Int) * p.strideKcsplit + (position p s).tmod p.blockSize * p.strideKcs + i.1.val).toNat

def copied {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl (fun t i =>
    t.write p.KCache (targetOffset p s i)
      (.mk .real (M.fp32Load ((s.mem p.K (sourceOffset p s.pids i)).read .real)))) s

def headState {α : Type} [Inhabited α] (p : Parameters) (s : State α) : State α :=
  let z := s.setReg "cur_token_idx" .int [] (fun _ => s.pids 0)
  let z := z.setReg "cur_seq_idx" .int [] (fun _ => sequence p s.pids)
  let z := z.setReg "cur_token_shift" .int [] (fun _ => (s.pids 0:Int)-p.n*(sequence p s.pids+1))
  let z := z.setReg "cur_kv_head_idx" .int [] (fun _ => s.pids 1)
  z.setReg "split_x_idx" .int [] (fun _ => s.pids 2)

def metadataState {α : Type} [Inhabited α] (p : Parameters) (s : State α) : State α :=
  let z := headState p s
  let z := z.setReg "past_kv_seq_len" .int [] (fun _ => position p s)
  let z := z.setReg "last_bt_block_idx" .int [] (fun _ => (position p s).tdiv p.blockSize)
  let z := z.setReg "block_table_ptr" .ptr [] (fun _ => (p.blockTables.cast,(sequence p s.pids*p.strideBts).toNat))
  let z := z.setReg "block_id" .int [] (fun _ => selectedBlock p s)
  let z := z.setReg "offset_last_block" .int [] (fun _ => (position p s).tmod p.blockSize)
  z.setReg "offsets_dmodel" .int [p.width] (fun i => (s.pids 2:Int)*p.width+i.1.val)

def preparedState {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  let z := metadataState p s
  let z := z.setReg "offsets_k" .int [p.width] (fun i => sourceOffset p s.pids i)
  let z := z.setReg "k" .real [p.width] (fun i => M.fp32Load ((s.mem p.K (sourceOffset p s.pids i)).read .real))
  z.setReg "offsets_kcache" .int [p.width] (fun i =>
    selectedBlock p s*p.strideKcb+(s.pids 1:Int)*p.strideKch+
      (s.pids 2:Int)*p.strideKcsplit+(position p s).tmod p.blockSize*p.strideKcs+i.1.val)

def finalState {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl (fun t i =>
    t.write p.KCache (targetOffset p s i)
      (.mk .real (M.fp32Load ((s.mem p.K (sourceOffset p s.pids i)).read .real)))) (preparedState M p s)

def headBody (p : Parameters) := (kernel Bool.false p).surfaceBody.take 5
def metadataBody (p : Parameters) := ((kernel Bool.false p).surfaceBody.drop 5).take 6
def tailBody (v : Bool) (p : Parameters) := (kernel v p).surfaceBody.drop 11

theorem decomposition (v : Bool) (p : Parameters) :
    kernel v p = .mk [p.K,p.blockTables.cast,p.lengths.cast] [p.KCache]
      (headBody p ++ metadataBody p ++ tailBody v p) := by cases v <;> rfl

theorem head_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    run M (headBody p) s=some (headState p s) := by
  simp [headBody,kernel,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,numeric,numericLt,bop,Region.cast,Option.bind,
    FP.Structural.eval_truncDiv,FP.Structural.eval_truncMod,headState,sequence,
    Broadcast.leftIndex,Broadcast.rightIndex]
  try dsimp only [bop,Broadcast.leftIndex,Broadcast.rightIndex]
  simp [integralFloorDiv,show ¬(s.pids 0:Int)<0 from by omega]
  rfl

theorem metadata_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    run M (metadataBody p) (headState p s)=some (metadataState p s) := by
  simp [metadataBody,kernel,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,numeric,numericLt,bop,Region.cast,Option.bind,
    FP.Structural.eval_truncDiv,FP.Structural.eval_truncMod,metadataState,headState,
    selectedBlock,position,sequence,Broadcast.leftIndex,Broadcast.rightIndex]
  try dsimp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.sameShape]
  simp [integralFloorDiv,←IntegerExpr.tdiv_formula,Int.tmod_def,
    Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right]
  congr 20 <;> funext <;> simp [bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem tail_run {α : Type} [Inhabited α] (M : Algebra α) (v : Bool)
    (p : Parameters) (s : State α) :
    run M (tailBody v p) (metadataState p s)=some (finalState M p s) := by
  cases v <;> simp [tailBody,kernel,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,store,numeric,bop,ComputeDType.eraseDType,
    Region.cast,Option.bind,finalState,preparedState,metadataState,headState,
    sourceOffset,targetOffset,Broadcast.leftIndex,Broadcast.rightIndex,
    Int.add_comm,Int.add_left_comm,Int.add_assoc]
  all_goals try dsimp only [bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  all_goals simp [Int.add_comm,Int.add_left_comm,Int.add_assoc,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]
  all_goals congr 20
  all_goals funext
  all_goals simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  all_goals simp [Int.add_comm,Int.add_left_comm,Int.add_assoc,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]

  all_goals first | ring | (congr 2 <;> ring)

/-- Both complete kernels execute successfully. The explicit witness records
register values, not a second source AST. -/
theorem execute {α : Type} [Inhabited α] (M : Algebra α) (v : Bool)
    (p : Parameters) (s : State α) :
    exec M (kernel v p) s = some (finalState M p s) := by
  rw [decomposition]
  simp only [FP.Structural.exec,run_append,head_run,Option.bind_some,metadata_run,tail_run]

def execution {α : Type} [Inhabited α] (M : Algebra α) (v : Bool)
    (p : Parameters) (s : State α) : {t : State α // exec M (kernel v p) s=some t} :=
  ⟨finalState M p s,execute M v p s⟩

theorem scatter_memory {α : Type} [Inhabited α] {I : Type} (xs : List I)
    (s : State α) (r : RegionName) (offset : I → Nat) (value : I → Cell α) :
    (xs.foldl (fun t i => t.write r (offset i) (value i)) s).mem =
      xs.foldl (fun m i => fun r' o => if r'=r ∧ o=offset i then value i else m r' o) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => exact ih (s.write r (offset i) (value i))

/-- The complete memory result is an independent snapshot gather/scatter,
including every untouched cell and arbitrary source/destination aliasing. -/


theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (v : Bool)
    (p : Parameters) (s : State α) :
    (execution M v p s).val.mem = (copied M p s).mem := by
  simp only [execution,finalState,copied,scatter_memory]
  rfl

theorem frame {α : Type} [Inhabited α] (M : Algebra α) (v : Bool)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat)
    (outside : r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], o ≠ targetOffset p s i) :
    (execution M v p s).val.mem r o = s.mem r o := by
  rw [execution_memory]
  apply State.scatter_frame
  exact outside.imp id (fun h i _ => h i)

#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean frame
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.KernelFP
