import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Loop

namespace VeriTile.Bench.Repairs.KldivOps.Program
open VeriTile.Triton FP.Structural FP.ProfiledRewrite Tile
noncomputable section
attribute [local irreducible] Tile.chunkRun Tile.prefixRun
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def writeMemory {α : Type} (mem : Memory α) (region : RegionName) (offset : Nat) (cell : Cell α) : Memory α :=
  fun r i => if r=region ∧ i=offset then cell else mem r i

def rowModel {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row : Nat) (eps : α) (mem : Memory α) : Memory α :=
  let result := loopModel M p flag red row eps 0 (mem,zero M)
  if red=0 then result.1 else writeMemory result.1 p.O row (.mk .real result.2)

def finishRun {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row : Nat) (eps carry : α) (s : State α) (h : Registers p red row eps carry s) :
    {final : State α // run (engine M) (forwardFinish p flag red) s=some final} := by
  cases flag <;> by_cases hr : red=0
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [forwardFinish,forwardTemplate,KernelDSL.forward,KernelDSL.forwardDefault,KernelDSL.forwardLog,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,Algebra.withDefaultPrecision,resolvePrecision,
      ComputeDType.eraseDType,State.setReg,numeric,numericEq,numericLt,natLt,
      FP.Structural.bop,store,Region.cast,hr,h.o,h.sum]
    rfl

theorem finish_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row : Nat) (eps carry : α) (s : State α) (h : Registers p red row eps carry s) :
    (finishRun M p flag red row eps carry s h).val.mem=
      if red=0 then s.mem else writeMemory s.mem p.O row (.mk .real carry) := by
  cases flag <;> by_cases hr : red=0
  all_goals simp [finishRun,hr,TileShape.allIndices,List.finRange_succ,State.write,writeMemory]
  all_goals rfl

attribute [irreducible] finishRun

/-- Actual whole row, including prefix, every chunk, masked final tile and
final scalar store. Exact FP values remain symbolic through the whole proof. -/
theorem row_complete {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red : Nat) (eps : α) (s : State α) (hb : 0<p.block) :
    ∃ final, run (engine M) (forward p flag red).surfaceBody (bindEpsilon eps s)=some final ∧
      final.mem=rowModel M p flag red (s.pids 0) eps s.mem := by
  let prepared := (prefixRun M p flag red eps s).val
  have hp := (prefixRun M p flag red eps s).property
  have hi := prefix_registers M p flag red eps s
  obtain ⟨looped,hl,hm,hr⟩ := loop_complete M p flag red (s.pids 0) eps 0 (zero M) prepared hi hb
  let carry := (loopModel M p flag red (s.pids 0) eps 0 (prepared.mem,zero M)).2
  let final := (finishRun M p flag red (s.pids 0) eps carry looped hr).val
  have hf := (finishRun M p flag red (s.pids 0) eps carry looped hr).property
  have hs : step (engine M) (.forRangeInt "i" (.constInt 0) (.constInt p.cols)
      (.constNat p.block) (forwardBody p flag red)) prepared=some looped := by
    simpa [step,evalOp_unfold] using hl
  have hloop : run (engine M) [.forRangeInt "i" (.constInt 0) (.constInt p.cols)
      (.constNat p.block) (forwardBody p flag red)] prepared=some looped := by
    simp [run,hs]
  refine ⟨final,?_,?_⟩
  · simpa [forward_decomposition,run_append,hp,run,hs,prepared] using hf
  · rw [finish_memory]
    simp only [hm,prefix_memory,carry,rowModel,prepared]

open VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms (runSchedule)

def gridModel {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red : Nat) (eps : α) (jobs : List Nat) (mem : Memory α) : Memory α :=
  jobs.foldl (fun m row => rowModel M p flag red row eps m) mem

def atRow {α : Type} (s : State α) (row : Nat) :=
  {s with pids:=fun ax => if ax=0 then row else 0}

def grid {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red : Nat) (eps : α) (s : State α) : Option (State α) :=
  runSchedule (fun row current => run (engine M) (forward p flag red).surfaceBody
    (bindEpsilon eps (atRow current row))) (List.range p.rows) s

theorem schedule_complete {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red : Nat) (eps : α) (jobs : List Nat) (s : State α) (hb : 0<p.block) :
    ∃ final, runSchedule (fun row current => run (engine M) (forward p flag red).surfaceBody
      (bindEpsilon eps (atRow current row))) jobs s=some final ∧
      final.mem=gridModel M p flag red eps jobs s.mem := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons row rest ih =>
    obtain ⟨next,he,hm⟩ := row_complete M p flag red eps (atRow s row) hb
    obtain ⟨final,hf,hmem⟩ := ih next
    refine ⟨final,?_,?_⟩
    · simp only [runSchedule,he,Option.bind_some]; exact hf
    · simpa only [gridModel,List.foldl_cons,hm,atRow,if_pos rfl] using hmem

theorem grid_complete {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red : Nat) (eps : α) (s : State α) (hb : 0<p.block) :
    ∃ final, grid M p flag red eps s=some final ∧
      final.mem=gridModel M p flag red eps (List.range p.rows) s.mem :=
  schedule_complete M p flag red eps _ s hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Program
