import bench.tritonbench_g.kldiv_ops.Proofs.Forward

namespace VeriTile.Bench.Repairs.KldivOps.Backward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open Program (Memory memoryState writeMemory atRow)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def backwardPrefix (p : Parameters) (flag : Bool) := (backward p flag).surfaceBody.take 5

def body (p : Parameters) (flag : Bool) : List ComputeStmt :=
  match ((backward p flag).surfaceBody[5]? : Option ComputeStmt) with
  | some (ComputeStmt.forRangeInt _ _ _ _ statements) => statements
  | _ => []

structure Registers {α : Type} (p : Parameters) (row : Nat) (s : State α) : Prop where
  target : s.regs .ptr [] "T"=some (fun _ => (p.T,row*p.cols))
  output : s.regs .ptr [] "O"=some (fun _ => (p.O,row*p.cols))

def scalar (M : Algebra α) (flag : Bool) (target : α) : α :=
  if flag then M.unary (some .fp32) (.neg .real) (M.unary (some .fp32) .exp (M.fp32Load target))
  else M.binary (some .fp32) .real .mul (M.fp32Load target) (M.literal (some .fp32) .real (-1))

def prefixRun {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool) (s : State α) :
    {t : State α // run (engine M) (backwardPrefix p flag) s=some t} := by
  cases flag
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [backwardPrefix,backward,KernelDSL.backward,KernelDSL.backwardDefault,KernelDSL.backwardLog,VeriTile.Bench.Repairs.KldivOps.BackwardSource.defaultTarget,VeriTile.Bench.Repairs.KldivOps.BackwardSource.logarithmic,VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLog,
      VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefault,
      VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefaultOriginal,VeriTile.Bench.Repairs.KldivOps.BackwardSource.defaultTarget,
      VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLogOriginal,VeriTile.Bench.Repairs.KldivOps.BackwardSource.logarithmic,

      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      State.setReg,numeric,numericLt,natLt,FP.Structural.bop,Region.cast]
    rfl

theorem prefix_registers {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool) (s : State α) :
    Registers p (s.pids 0) (prefixRun M p flag s).val := by
  cases flag <;> constructor <;> simp [prefixRun,State.setReg,FP.Structural.bop,Region.cast,Int.toNat_mul]
  all_goals funext i; simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

def chunkRun {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (row start : Nat) (s : State α) (h : Registers p row s) :
    {t : State α // run (engine M) (body p flag)
      (s.setReg "i" .int [] (fun _ => (start : Int)))=some t} := by
  cases flag
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [body,backward,KernelDSL.backward,KernelDSL.backwardDefault,KernelDSL.backwardLog,VeriTile.Bench.Repairs.KldivOps.BackwardSource.defaultTarget,VeriTile.Bench.Repairs.KldivOps.BackwardSource.logarithmic,VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLog,
      VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefault,
      VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefaultOriginal,VeriTile.Bench.Repairs.KldivOps.BackwardSource.defaultTarget,
      VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLogOriginal,VeriTile.Bench.Repairs.KldivOps.BackwardSource.logarithmic,

      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      State.setReg,numeric,numericLt,natLt,FP.Structural.bop,store,Region.cast,h.target,h.output]
    rfl

def chunkMemory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (row start : Nat) (mem : Memory α) : Memory α :=
  (TileShape.allIndices [p.block]).foldl (fun current j =>
    if start+j.1.val<p.cols then writeMemory current p.O (row*p.cols+(start+j.1.val))
      (.mk .real (scalar M flag ((mem p.T (row*p.cols+(start+j.1.val))).read .real)))
    else current) mem

theorem chunk_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (row start : Nat) (s : State α) (h : Registers p row s) :
    (chunkRun M p flag row start s h).val.mem=chunkMemory M p flag row start s.mem := by
  cases flag
  all_goals
    simp [chunkRun,chunkMemory,scalar,writeMemory,Tile.zero,
      VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory,
      State.setReg,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,Int.toNat_add,Tile.active_int,ofFloat,toFloat,engine,
      Algebra.withDefaultPrecision,resolvePrecision]
  all_goals norm_num
  all_goals
    congr 1
    funext current j
    by_cases active : start+j.1.val<p.cols
    all_goals simp [active,Program.writeMemory]
    all_goals rfl

theorem chunk_registers {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (row start : Nat) (s : State α) (h : Registers p row s) :
    Registers p row (chunkRun M p flag row start s h).val := by
  cases flag <;> constructor
  all_goals simp [chunkRun,Tile.masked_regs,State.setReg,h.target,h.output]

attribute [irreducible] chunkRun prefixRun


def loopModel {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (row start : Nat) (mem : Memory α) : Memory α :=
  if h : 0<p.block ∧ start<p.cols then
    loopModel M p flag row (start+p.block) (chunkMemory M p flag row start mem)
  else mem
termination_by p.cols-start
decreasing_by omega

theorem loop_complete {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (row start : Nat) (s : State α) (h : Registers p row s) (hb : 0<p.block) :
    ∃ final, rangeInt (engine M) "i" (start : Int) (p.cols : Int) p.block (body p flag) s=some final ∧
      final.mem=loopModel M p flag row start s.mem ∧ Registers p row final := by
  by_cases active : start<p.cols
  · let next := (chunkRun M p flag row start s h).val
    have he := (chunkRun M p flag row start s h).property
    have hm := chunk_memory M p flag row start s h
    have hr := chunk_registers M p flag row start s h
    obtain ⟨final,hf,hmem,hregs⟩ := loop_complete M p flag row (start+p.block) next hr hb
    refine ⟨final,?_,?_,hregs⟩
    · rw [rangeInt,if_neg (by omega),if_pos (by exact_mod_cast active),he]
      simpa only [Option.bind_some,←Int.natCast_add] using hf
    · rw [loopModel,dif_pos ⟨hb,active⟩]
      simpa only [next,hm] using hmem
  · refine ⟨s,?_,?_,h⟩
    · rw [rangeInt,if_neg (by omega),if_neg (by exact_mod_cast active)]
    · rw [loopModel,dif_neg (by tauto)]
termination_by p.cols-start
decreasing_by omega

theorem source_decomposition (p : Parameters) (flag : Bool) :
    (backward p flag).surfaceBody=backwardPrefix p flag ++
      [.forRangeInt "i" (.constInt 0) (.constInt p.cols) (.constNat p.block) (body p flag)] := by
  cases flag <;> rfl

theorem row_complete {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (s : State α) (hb : 0<p.block) :
    ∃ final, run (engine M) (backward p flag).surfaceBody s=some final ∧
      final.mem=loopModel M p flag (s.pids 0) 0 s.mem := by
  let prepared := (prefixRun M p flag s).val
  have hp := (prefixRun M p flag s).property
  have hi := prefix_registers M p flag s
  have hpMem : prepared.mem=s.mem := by cases flag <;> unfold prepared prefixRun <;> rfl
  obtain ⟨final,he,hm,hr⟩ := loop_complete M p flag (s.pids 0) 0 prepared hi hb
  have hs : step (engine M) (.forRangeInt "i" (.constInt 0) (.constInt p.cols)
      (.constNat p.block) (body p flag)) prepared=some final := by
    simpa [step,evalOp_unfold] using he
  refine ⟨final,?_,?_⟩
  · rw [source_decomposition,run_append,hp]
    simpa [run,hs,prepared]
  · simpa only [hpMem] using hm

open VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms (runSchedule)

def gridModel {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (jobs : List Nat) (mem : Memory α) : Memory α :=
  jobs.foldl (fun current row => loopModel M p flag row 0 current) mem

def grid {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (s : State α) : Option (State α) :=
  runSchedule (fun row current => run (engine M) (backward p flag).surfaceBody (atRow current row))
    (List.range p.rows) s

theorem schedule_complete {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (jobs : List Nat) (s : State α) (hb : 0<p.block) :
    ∃ final, runSchedule (fun row current => run (engine M) (backward p flag).surfaceBody
      (atRow current row)) jobs s=some final ∧ final.mem=gridModel M p flag jobs s.mem := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons row rest ih =>
    obtain ⟨next,he,hm⟩ := row_complete M p flag (atRow s row) hb
    obtain ⟨final,hf,hmem⟩ := ih next
    refine ⟨final,?_,?_⟩
    · simp only [runSchedule,he,Option.bind_some]; exact hf
    · simpa only [gridModel,List.foldl_cons,hm,atRow,if_pos rfl] using hmem

theorem grid_complete {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (s : State α) (hb : 0<p.block) :
    ∃ final, grid M p flag s=some final ∧ final.mem=gridModel M p flag (List.range p.rows) s.mem :=
  schedule_complete M p flag _ s hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Backward
