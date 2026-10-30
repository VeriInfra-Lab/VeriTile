import bench.tritonbench_g.chunk_linear_attn.Model
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure StateReady (pid : PID) (input : Memory α) (s : State α) : Prop where
  inputs : StateInputs s.mem input
  ik : s.regs .nat [] "i_k"=some (fun _ => pid.x)
  iv : s.regs .nat [] "i_v"=some (fun _ => pid.y)
  bh : s.regs .nat [] "i_bh"=some (fun _ => pid.z)
def HasCarry (p : Parameters) (rev : Bool) (h : HValues p α) (s : State α) :=
  s.regs .real [p.BK,p.BV] (carryName rev)=some h

def stateEntry (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (s : State α) :
    {t : State α // run (engine (withDot M D)) (stateParts p scale rev seed final).before s=some t} := by
  cases rev <;> cases seed
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [stateParts,stateKernel,loopName,forwardH,backwardH,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
    rfl

theorem stateEntry_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (s : State α) :
    (stateEntry M D p scale rev seed final s).val.mem=s.mem := by
  cases rev <;> cases seed <;> simp [stateEntry]

theorem stateEntry_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (s : State α) :
    let pid : PID := ⟨s.pids 0,s.pids 1,s.pids 2⟩
    StateReady pid s.mem (stateEntry M D p scale rev seed final s).val ∧
    HasCarry p rev (stateInitial M p pid s.mem rev seed) (stateEntry M D p scale rev seed final s).val := by
  dsimp only
  constructor
  · constructor
    · rw [stateEntry_memory]
      exact StateInputs.refl _
    · cases rev <;> cases seed <;> simp [stateEntry]
    · cases rev <;> cases seed <;> simp [stateEntry]
    · cases rev <;> cases seed <;> simp [stateEntry]
  · cases rev <;> cases seed <;>
      simp [HasCarry,carryName,stateEntry,VeriTile.Bench.Repairs.ChunkLinearAttn.stateInitial,seedName,rawLoad,zero,cast,
        FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals
      funext i
      simp [VeriTile.Bench.Repairs.ChunkLinearAttn.stateInitial,rawLoad,seedName,cast,zero]

#axiomsClean stateEntry_memory
#axiomsClean stateEntry_ready
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
