import bench.tritonbench_g.chunk_linear_attn.Proofs.Support.StateEntry
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The reached Python chunk index, with its actual source carrier. -/
def StateTime (p : Parameters) (rev : Bool) (n : Nat) (s : State α) : Prop :=
  if rev then s.regs .int [] "i_t"=some (fun _ => (chunk p rev n : Int))
  else s.regs .nat [] "i_t"=some (fun _ => chunk p rev n)

theorem reverse_clock (p : Parameters) (n : Nat) (hn : n<p.NT) :
    (0 : Int)-((n : Int)+1-(p.NT : Int))=((p.NT-1-n : Nat) : Int) := by omega

theorem natural_history_address (a b c d e : Nat) :
    Int.toNat ((a : Int)*((b : Int)*c*d)+(e : Int)*c*d)=a*(b*c*d)+e*c*d := by norm_cast

theorem natural_offset (a b : Nat) : Int.toNat ((a : Int)*b)=a*b := by norm_cast

def stateStoreExecution (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s) :
    {t : State α // run (engine (withDot M D)) (stateStorePrefix p scale rev seed final)
      (stateIndex p rev n s)=some t} := by
  cases rev
  all_goals
    simp only [HasCarry,carryName,Bool.false_eq_true,if_false,if_true] at hh
    refine ⟨?_,?_⟩
    swap
    simp [stateStorePrefix,stateParts,stateIntParts,stateIndex,stateKernel,loopName,forwardH,forwardHKernel,backwardH,backwardHKernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Source.intRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,FP.Structural.store,ctx.ik,ctx.iv,ctx.bh,hh]
    rfl

theorem stateStore_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s) (hn : n<p.NT) :
    (stateStoreExecution M D p scale rev seed final pid input s n h ctx hh).val.mem=
      writeHistory M p pid rev n h s.mem := by
  have clock : (p.NT : Int)-((n : Int)+1)=((p.NT-1-n : Nat) : Int) := by omega
  cases rev <;> simp [stateStoreExecution,writeHistory,scatter,masked_scatter_memory,hActive,hOffset,
    historyName,chunk,cast,clock,natural_history_address,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rfl

theorem stateStore_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s) (hn : n<p.NT) :
    let t := (stateStoreExecution M D p scale rev seed final pid input s n h ctx hh).val
    StateReady pid input t ∧ HasCarry p rev h t ∧
      StateTime p rev n t := by
  dsimp only
  refine ⟨?_,?_,?_⟩
  · constructor
    · rw [stateStore_memory M D p scale rev seed final pid input s n h ctx hh hn]
      exact (history_inputs _ _ _ _ _ _ _).trans ctx.inputs
    · cases rev <;> simpa [stateStoreExecution] using ctx.ik
    · cases rev <;> simpa [stateStoreExecution] using ctx.iv
    · cases rev <;> simpa [stateStoreExecution] using ctx.bh
  · cases rev <;> simpa [HasCarry,carryName,stateStoreExecution] using hh
  · have clock : (p.NT : Int)-((n : Int)+1)=((p.NT-1-n : Nat) : Int) := by omega
    cases rev <;> simp [StateTime,stateStoreExecution,chunk,clock,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    funext i
    change (0 : Int)-((n : Int)+1-(p.NT : Int))=((p.NT-1-n : Nat) : Int)
    exact reverse_clock p n hn

structure StatePointers (p : Parameters) (pid : PID) (rev : Bool) (n : Nat) (s : State α) : Prop where
  left : s.regs .blockPtr [p.BK,p.BT] (if rev then "p_q" else "p_k")=
    some (fun _ => BlockPtr.mk (if rev then "query" else "key") (pid.z*(p.T*p.K))
      [p.K,p.T] [p.BK,p.BT] [1,p.K] [pid.x*p.BK,chunk p rev n*p.BT])
  right : s.regs .blockPtr [p.BT,p.BV] (if rev then "p_do" else "p_v")=
    some (fun _ => BlockPtr.mk (if rev then "incoming" else "value") (pid.z*(p.T*p.V))
      [p.T,p.V] [p.BT,p.BV] [p.V,1] [chunk p rev n*p.BT,pid.y*p.BV])

theorem stateStore_pointers (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s) (hn : n<p.NT) :
    StatePointers p pid rev n (stateStoreExecution M D p scale rev seed final pid input s n h ctx hh).val := by
  have clock : (p.NT : Int)-((n : Int)+1)=((p.NT-1-n : Nat) : Int) := by omega
  constructor <;> cases rev <;>
    simp [stateStoreExecution,chunk,clock,natural_offset,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

def stateUpdateExecution (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s)
    (ht : StateTime p rev n s) (hp : StatePointers p pid rev n s) :
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
    simp only [StateTime,Bool.false_eq_true,if_false,if_true] at ht
    simp only [Bool.false_eq_true,if_false,if_true] at hl hr
    refine ⟨?_,?_⟩
    swap
    simp [stateUpdateSuffix,stateParts,stateIntParts,stateIndex,stateKernel,loopName,forwardH,forwardHKernel,backwardH,backwardHKernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Source.intRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,ctx.ik,ctx.iv,ctx.bh,hh,ht,hq,hk,hv,hg,hl,hr]
    rfl

theorem stateUpdate_values (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s)
    (ht : StateTime p rev n s) (hp : StatePointers p pid rev n s) :
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
