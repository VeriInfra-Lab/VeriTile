import bench.tritonbench_g.chunk_retention_ops.Sources.KernelBackwardStore
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
namespace L
export VeriTile.Bench.Repairs.ChunkLinearAttn (zero scalar cast add mul dot rawLoad load)
end L

def factor (M : Algebra α) (p : Parameters) (pid : PID) (n : Nat) : α :=
  M.unary (some .fp32) .exp2 (L.mul M (M.fromNat (some .fp32)
    (min p.BT (p.T-time p n*p.BT))) (logRate M p pid))

def leftValues (M : Algebra α) (p : Parameters) (pid : PID) (input : Memory α)
    (scale : ℝ) (n : Nat) : Matrix α p.BK p.BT := fun i=>
  L.cast M (L.mul M (L.load M input "q" (pid.z*(p.T*p.K)) p.K p.T 1 p.K
    (pid.x*p.BK) (time p n*p.BT) i) (L.scalar M scale))
def rightValues (M : Algebra α) (p : Parameters) (pid : PID) (input : Memory α)
    (n : Nat) : Matrix α p.BT p.BV := fun i=>
  L.cast M (L.mul M (L.load M input "incoming" (pid.z*(p.T*p.V)) p.T p.V p.V 1
    (time p n*p.BT) (pid.y*p.BV) i) (weight M p pid i.1.val))
def next (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input : Memory α) (scale : ℝ) (n : Nat) (carry : Carry p α) : Carry p α := fun i=>
  L.add M (L.mul M (factor M p pid n) (carry i))
    (L.dot D (leftValues M p pid input scale n) (rightValues M p pid input n) i)

def updateExecution (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (terminal initial : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (carry : Carry p α) (ctx : Ready M p pid input s) (hc : HasCarry p carry s)
    (ht : s.regs .nat [] "i_t"=some (fun _=>time p n)) (hp : Pointers p pid n s) :
    {t : State α // run (engine (withDot M D)) (updateSuffix p scale terminal initial) s=some t} := by
  have hq:=ctx.inputs "q" (by simp)
  have hg:=ctx.inputs "incoming" (by simp)
  have hl:=hp.left
  have hr:=hp.right
  simp only [HasCarry] at hc
  refine ⟨?_,?_⟩
  swap
  simp [updateSuffix,parts,kernel,Source.backwardH,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
    ctx.ik,ctx.iv,ctx.bh,ctx.rate,ctx.lane,ctx.weights,hc,ht,hq,hg,hl,hr]
  rfl

theorem update_values (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (terminal initial : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (carry : Carry p α) (ctx : Ready M p pid input s) (hc : HasCarry p carry s)
    (ht : s.regs .nat [] "i_t"=some (fun _=>time p n)) (hp : Pointers p pid n s) :
    let t:=(updateExecution M D p scale terminal initial pid input s n carry ctx hc ht hp).val
    Ready M p pid input t ∧ HasCarry p (next M D p pid input scale n carry) t ∧ t.mem=s.mem := by
  dsimp only
  refine ⟨?_,?_,?_⟩
  · constructor
    · simpa [updateExecution] using ctx.inputs
    · simpa [updateExecution] using ctx.ik
    · simpa [updateExecution] using ctx.iv
    · simpa [updateExecution] using ctx.bh
    · simpa [updateExecution] using ctx.rate
    · simpa [updateExecution] using ctx.lane
    · simpa [updateExecution] using ctx.weights
  · simp [HasCarry,updateExecution,next,leftValues,rightValues,factor,L.load,L.rawLoad,
      L.zero,L.cast,L.add,L.mul,L.scalar,L.dot,time,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,Function.comp_def]
    funext i
    unfold next leftValues rightValues factor
    unfold VeriTile.Bench.Repairs.ChunkLinearAttn.load VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad
    have hm : (if p.BT<p.T-(p.NT-1-n)*p.BT then p.BT else p.T-(p.NT-1-n)*p.BT)=
        min p.BT (p.T-(p.NT-1-n)*p.BT) := by split_ifs <;> omega
    simp [L.zero,L.cast,L.add,L.mul,L.scalar,L.dot,time,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,hm]
  · simp [updateExecution]

#axiomsClean update_values
#specNonCircular next avoiding [run,FP.Structural.exec,entry,storeExecution,updateExecution]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
