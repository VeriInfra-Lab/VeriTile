import bench.tritonbench_g.chunk_retention_ops.Sources.KernelBackwardModel
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (zero scalar cast add mul dot rawLoad load scatter scatter_frame hActive hOffset)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def time (p : Parameters) (n : Nat) := p.NT-1-n

def writeHistory (M : Algebra α) (p : Parameters) (pid : PID) (n : Nat)
    (carry : Carry p α) (memory : Memory α) : Memory α :=
  scatter (TileShape.allIndices [p.BK,p.BV]) "dhistory" (hActive p pid)
    (hOffset p pid (time p n)) (fun i=>.mk .real (cast M (carry i))) memory

theorem history_inputs (M : Algebra α) (p : Parameters) (pid : PID) (n : Nat)
    (carry : Carry p α) (memory : Memory α) : InputEq (writeHistory M p pid n carry memory) memory := by
  intro r hr o
  apply scatter_frame
  left
  intro he
  subst r
  simp at hr

def storePrefix (p : Parameters) (scale : ℝ) (terminal initial : Bool) :=
  (parts p scale terminal initial).body.take 5

def updateSuffix (p : Parameters) (scale : ℝ) (terminal initial : Bool) :=
  (parts p scale terminal initial).body.drop 5

def storeExecution (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (terminal initial : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (carry : Carry p α) (ctx : Ready M p pid input s) (hc : HasCarry p carry s) :
    {t : State α // run (engine (withDot M D)) (storePrefix p scale terminal initial)
      (s.setReg "j" .nat [] (fun _=>n))=some t} := by
  simp only [HasCarry] at hc
  refine ⟨?_,?_⟩
  swap
  simp [storePrefix,parts,kernel,Source.backwardH,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,FP.Structural.store,
    ctx.ik,ctx.iv,ctx.bh,hc]
  rfl

theorem store_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (terminal initial : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (carry : Carry p α) (ctx : Ready M p pid input s) (hc : HasCarry p carry s) :
    (storeExecution M D p scale terminal initial pid input s n carry ctx hc).val.mem=
      writeHistory M p pid n carry s.mem := by
  simp [storeExecution,writeHistory,scatter,masked_scatter_memory,hActive,hOffset,time,VeriTile.Bench.Repairs.ChunkLinearAttn.cast,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

theorem store_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (terminal initial : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (carry : Carry p α) (ctx : Ready M p pid input s) (hc : HasCarry p carry s) :
    let t:=(storeExecution M D p scale terminal initial pid input s n carry ctx hc).val
    Ready M p pid input t ∧ HasCarry p carry t ∧
      t.regs .nat [] "i_t"=some (fun _=>time p n) := by
  dsimp only
  refine ⟨?_,?_,?_⟩
  · constructor
    · rw [store_memory]
      intro r hr o
      exact (history_inputs M p pid n carry s.mem r hr o).trans (ctx.inputs r hr o)
    · simpa [storeExecution] using ctx.ik
    · simpa [storeExecution] using ctx.iv
    · simpa [storeExecution] using ctx.bh
    · simpa [storeExecution] using ctx.rate
    · simpa [storeExecution] using ctx.lane
    · simpa [storeExecution] using ctx.weights
  · simpa [HasCarry,storeExecution] using hc
  · simp [storeExecution,time,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    rfl

structure Pointers (p : Parameters) (pid : PID) (n : Nat) (s : State α) : Prop where
  left : s.regs .blockPtr [p.BK,p.BT] "p_q"=some (fun _=>BlockPtr.mk "q"
    (pid.z*(p.T*p.K)) [p.K,p.T] [p.BK,p.BT] [1,p.K] [pid.x*p.BK,time p n*p.BT])
  right : s.regs .blockPtr [p.BT,p.BV] "p_do"=some (fun _=>BlockPtr.mk "incoming"
    (pid.z*(p.T*p.V)) [p.T,p.V] [p.BT,p.BV] [p.V,1] [time p n*p.BT,pid.y*p.BV])

theorem store_pointers (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (terminal initial : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (carry : Carry p α) (ctx : Ready M p pid input s) (hc : HasCarry p carry s) :
    Pointers p pid n (storeExecution M D p scale terminal initial pid input s n carry ctx hc).val := by
  constructor <;> simp [storeExecution,time,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals funext i
  all_goals congr 1

#axiomsClean store_memory
#axiomsClean store_ready
#axiomsClean store_pointers
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
