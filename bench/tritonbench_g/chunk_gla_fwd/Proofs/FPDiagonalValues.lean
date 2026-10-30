import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPDiagonal
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPDiagonal
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active masked_scatter_regs)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def active (split : Bool) (c : Layout) (s : State α) (i : Index c) : Prop := row split c s i.1.val<c.T
instance (split : Bool) (c : Layout) (s : State α) : DecidablePred (active split c s) :=
  fun _ => inferInstanceAs (Decidable (_<_))

def columnWritten (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ) (initial : State α)
    (j : Nat) (current : State α) : State α :=
  (TileShape.allIndices [c.BC]).foldl (fun t i =>
    if active split c initial i then
      t.write (out split) (outputAddress split c initial j i) (.mk .real (value M split c scale initial j i)) else t) current

theorem column_memory (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ) (initial : State α)
    (j : Nat) (s t : State α) (he : s.mem=t.mem) :
    (columnWritten M split c scale initial j s).mem=(columnWritten M split c scale initial j t).mem := by
  simp only [columnWritten,masked_scatter_memory,he]

theorem bodyMemory (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (initial current : State α) (j : Nat) (ctx : Context M split c initial current j) :
    (bodyExecution M split c scale initial current j ctx).val.mem=
      (columnWritten M split c scale initial j current).mem := by
  cases split
  all_goals simp [bodyExecution,columnWritten,masked_scatter_memory,active,outputAddress,value,product,keyLoad,queryLoad,zero,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
  all_goals norm_num only
  all_goals rfl

theorem columnFrame (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ) (initial current : State α)
    (j : Nat) (r : RegionName) (off : Nat)
    (hn : r≠out split ∨ ∀ i, active split c initial i → off≠outputAddress split c initial j i) :
    (columnWritten M split c scale initial j current).mem r off=current.mem r off := by
  apply State.masked_scatter_frame
  rcases hn with hr|ho
  · exact Or.inl hr
  · right
    intro i _ hi
    exact ho i hi

theorem bodyContext (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (initial current : State α) (j : Nat) (ctx : Context M split c initial current j) :
    Context M split c initial (bodyExecution M split c scale initial current j ctx).val (j+1) := by
  constructor
  · funext off
    rw [bodyMemory,columnFrame M split c scale initial current j "k" off (Or.inl (by cases split <;> decide))]
    exact congrFun ctx.keyMemory off
  · funext off
    rw [bodyMemory,columnFrame M split c scale initial current j "g" off (Or.inl (by cases split <;> decide))]
    exact congrFun ctx.gateMemory off
  all_goals cases split
  all_goals simp [bodyExecution,ctx.query,ctx.gate,ctx.key,ctx.gateKey,ctx.featureMask,ctx.rowMask,ctx.offsets,ctx.rowIndex,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,keyAddress,Nat.add_mul,Nat.add_assoc]
  all_goals
    funext i
    simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,keyAddress,Nat.add_mul,Nat.add_assoc]

#axiomsClean bodyMemory
#axiomsClean bodyContext
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPDiagonal
