import bench.repairs.tritonbench_g.chunk_linear_attn.StateEntry
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def stateStoreExecution (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s) :
    {t : State α // run (engine (withDot M D)) (stateStorePrefix p scale rev seed final)
      (s.setReg (loopName rev) .nat [] (fun _ => n))=some t} := by
  cases rev
  all_goals
    simp only [HasCarry,carryName,Bool.false_eq_true,if_false,if_true] at hh
    refine ⟨?_,?_⟩
    swap
    simp [stateStorePrefix,stateParts,stateKernel,loopName,forwardH,backwardH,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,FP.Structural.store,ctx.ik,ctx.iv,ctx.bh,hh]
    rfl

theorem stateStore_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s) :
    (stateStoreExecution M D p scale rev seed final pid input s n h ctx hh).val.mem=
      writeHistory M p pid rev n h s.mem := by
  cases rev <;> simp [stateStoreExecution,writeHistory,scatter,masked_scatter_memory,hActive,hOffset,
    historyName,chunk,cast,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rfl

theorem stateStore_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s) :
    let t := (stateStoreExecution M D p scale rev seed final pid input s n h ctx hh).val
    StateReady pid input t ∧ HasCarry p rev h t ∧
      t.regs .nat [] "i_t"=some (fun _ => chunk p rev n) := by
  dsimp only
  refine ⟨?_,?_,?_⟩
  · constructor
    · rw [stateStore_memory]
      exact (history_inputs _ _ _ _ _ _ _).trans ctx.inputs
    · cases rev <;> simpa [stateStoreExecution] using ctx.ik
    · cases rev <;> simpa [stateStoreExecution] using ctx.iv
    · cases rev <;> simpa [stateStoreExecution] using ctx.bh
  · cases rev <;> simpa [HasCarry,carryName,stateStoreExecution] using hh
  · cases rev <;> simp [stateStoreExecution,chunk,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex] <;> rfl

structure StatePointers (p : Parameters) (pid : PID) (rev : Bool) (n : Nat) (s : State α) : Prop where
  left : s.regs .blockPtr [p.BK,p.BT] (if rev then "p_q" else "p_k")=
    some (fun _ => BlockPtr.mk (if rev then "query" else "key") (pid.z*(p.T*p.K))
      [p.K,p.T] [p.BK,p.BT] [1,p.K] [pid.x*p.BK,chunk p rev n*p.BT])
  right : s.regs .blockPtr [p.BT,p.BV] (if rev then "p_do" else "p_v")=
    some (fun _ => BlockPtr.mk (if rev then "incoming" else "value") (pid.z*(p.T*p.V))
      [p.T,p.V] [p.BT,p.BV] [p.V,1] [chunk p rev n*p.BT,pid.y*p.BV])

theorem stateStore_pointers (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s) :
    StatePointers p pid rev n (stateStoreExecution M D p scale rev seed final pid input s n h ctx hh).val := by
  constructor <;> cases rev <;>
    simp [stateStoreExecution,chunk,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

def stateUpdateExecution (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s)
    (ht : s.regs .nat [] "i_t"=some (fun _ => chunk p rev n)) (hp : StatePointers p pid rev n s) :
    {t : State α // run (engine (withDot M D)) (stateUpdateSuffix p scale rev seed final) s=some t} := by
  have hq := ctx.inputs "query" (by simp)
  have hk := ctx.inputs "key" (by simp)
  have hv := ctx.inputs "value" (by simp)
  have hg := ctx.inputs "incoming" (by simp)
  have hl := hp.left
  have hr := hp.right
  cases rev
  all_goals
    simp only [HasCarry,carryName,Bool.false_eq_true,if_false,if_true] at hh
    simp only [Bool.false_eq_true,if_false,if_true] at hl hr
    refine ⟨?_,?_⟩
    swap
    simp [stateUpdateSuffix,stateParts,stateKernel,loopName,forwardH,backwardH,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,ctx.ik,ctx.iv,ctx.bh,hh,ht,hq,hk,hv,hg,hl,hr]
    rfl

theorem stateUpdate_values (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s)
    (ht : s.regs .nat [] "i_t"=some (fun _ => chunk p rev n)) (hp : StatePointers p pid rev n s) :
    let t := (stateUpdateExecution M D p scale rev seed final pid input s n h ctx hh ht hp).val
    StateReady pid input t ∧ HasCarry p rev (stateNext M D p pid input scale rev n h) t ∧ t.mem=s.mem := by
  dsimp only
  refine ⟨?_,?_,?_⟩
  · constructor
    · cases rev <;> simpa [stateUpdateExecution] using ctx.inputs
    · cases rev <;> simpa [stateUpdateExecution] using ctx.ik
    · cases rev <;> simpa [stateUpdateExecution] using ctx.iv
    · cases rev <;> simpa [stateUpdateExecution] using ctx.bh
  · cases rev <;>
      simp [HasCarry,carryName,stateUpdateExecution,stateNext,stateLeft,stateRight,load,rawLoad,
        zero,cast,add,mul,scalar,dot,chunk,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals
      funext i
      dsimp [stateNext,stateLeft,stateRight,load,rawLoad,zero,cast,add,mul,scalar,dot,chunk]
      unfold VeriTile.Bench.Repairs.ChunkLinearAttn.stateLeft VeriTile.Bench.Repairs.ChunkLinearAttn.stateRight
      unfold VeriTile.Bench.Repairs.ChunkLinearAttn.load VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad
      simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,chunk,cast,mul,scalar,zero]
  · cases rev <;> simp [stateUpdateExecution]

#axiomsClean stateStore_memory
#axiomsClean stateStore_ready
#axiomsClean stateUpdate_values
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
