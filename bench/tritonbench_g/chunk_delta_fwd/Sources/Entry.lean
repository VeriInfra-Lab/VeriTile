import bench.tritonbench_g.chunk_delta_fwd.Sources.Model
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Ready (pid : PID) (input : Memory α) (s : State α) : Prop where
  inputs : SameInputs s.mem input
  ik : s.regs .nat [] "i_k"=some (fun _ => 0)
  iv : s.regs .nat [] "i_v"=some (fun _ => pid.1)
  bh : s.regs .nat [] "i_bh"=some (fun _ => pid.2)
def HasH (p : Parameters) (h : HValues p α) (s : State α) := s.regs .real [p.BK,p.BV] "b_h"=some h
def HasAcc (p : Parameters) (h : HValues p α) (s : State α) := s.regs .real [p.BK,p.BV] "b_h_cumsum"=some h

def entryState (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  let base := (((s.setReg "i_k" .nat [] (fun _ => s.pids 0)).setReg "i_v" .nat [] (fun _ => s.pids 1)).setReg "i_bh" .nat [] (fun _ => s.pids 2)).setReg "b_h" .real [p.BK,p.BV] (fun _ => zero M)
  if p.useSeed then
    let pointer := BlockPtr.mk "initial" (s.pids 2*p.K*p.V) [p.K,p.V] [p.BK,p.BV] [p.V,1] [s.pids 0*p.BK,s.pids 1*p.BV]
    (base.setReg "p_h0" .blockPtr [p.BK,p.BV] (fun _ => pointer)).setReg "b_h" .real [p.BK,p.BV]
      (fun i => cast M (if s.pids 0*p.BK+i.1.val<p.K ∧ s.pids 1*p.BV+i.2.1.val<p.V then
        (s.mem "initial" (s.pids 2*p.K*p.V+(s.pids 0*p.BK+i.1.val)*p.V+(s.pids 1*p.BV+i.2.1.val))).read .real else zero M))
  else base

theorem entry_run (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α) :
    run (engine (withDot M D)) (outer p).before s=some (entryState M p s) := by
  cases hp : p.useSeed <;>
     simp [outer,kernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
       run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
       Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
       numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
       TileShape.axisDim,Region.cast,State.setReg,hp,entryState,zero,cast,
       BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]

def entryExecution (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α) :
    {t : State α // run (engine (withDot M D)) (outer p).before s=some t} :=
  ⟨entryState M p s,entry_run M D p s⟩

theorem entry_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α) :
    (entryExecution M D p s).val.mem=s.mem := by
  cases hp : p.useSeed <;> simp [entryExecution,entryState,hp]

theorem entry_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α)
    (hk : s.pids 0=0) :
    Ready (s.pids 1,s.pids 2) s.mem (entryExecution M D p s).val ∧
    HasH p (initial (α:=α) M p (s.pids 1,s.pids 2) s.mem) (entryExecution M D p s).val := by
  constructor
  · constructor
    · intro r hr o
      rw [entry_memory]
    · cases hp : p.useSeed <;> simp [entryExecution,entryState,hp,hk]
    · cases hp : p.useSeed <;> simp [entryExecution,entryState,hp]
    · cases hp : p.useSeed <;> simp [entryExecution,entryState,hp]
  · cases hp : p.useSeed <;>
      simp [HasH,entryExecution,entryState,hp,VeriTile.Bench.Repairs.ChunkDeltaFwd.initial,zero,cast,column,hk,FP.Structural.bop,
        Broadcast.leftIndex,Broadcast.rightIndex,BlockPtr.inBounds_2d_offsets,
        BlockPtr.address_2d_offsets,TileShape.indexToList]
    all_goals
      funext i
      simp [initial,hp,zero,cast,column]

#axiomsClean entry_memory
#axiomsClean entry_ready
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd
