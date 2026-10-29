import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelStep
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReference

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelReverseStep
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelOps KernelStep
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def value (M : Algebra α) (incoming : Fin 64 → Fin 16 → Nat) (gate : Nat)
    (carry : Lane → α) (u : State α) : Lane → α :=
  next M carry (loadedScalar M "D" gate u) (loadedMatrix M "DS" incoming u)
def saved (source : Fin 64 → Fin 16 → Nat) (u : State α) : Lane → α := raw "S" .fp32 source u
def contraction (M : Algebra α) (source incoming : Fin 64 → Fin 16 → Nat) (gate : Nat)
    (carry : Lane → α) (u : State α) : α :=
  KernelReference.contraction M (value M incoming gate carry u) (saved source u)
def pre (M : Algebra α) (source incoming : Fin 64 → Fin 16 → Nat) (gate : Nat)
    (carry : Lane → α) (u : State α) : State α :=
  let v := u.setReg "S_i" .fp32 [64,16] (saved source u)
  let v := v.setReg "DS_i" .real [64,16] (loadedMatrix M "DS" incoming u)
  let v := v.setReg "d_i" .real [] (fun _ => loadedScalar M "D" gate u)
  let v := v.setReg "Dacc" .real [64,16] (value M incoming gate carry u)
  v.setReg "DG_i" .real [] (fun _ => contraction M source incoming gate carry u)
def written (M : Algebra α) (source incoming : Fin 64 → Fin 16 → Nat) (gate gradient : Nat)
    (carry : Lane → α) (u : State α) : State α :=
  let v := (pre M source incoming gate carry u).write "DG" gradient
    (KernelReference.stored M .real (contraction M source incoming gate carry u))
  scatter "DI" .real source
    (fun i => M.cast (some .fp32) .real .real (value M incoming gate carry u i)) v
def result (M : Algebra α) (c : Shape) (source incoming : Fin 64 → Fin 16 → Nat)
    (gate gradient : Nat) (carry : Lane → α) (u : State α) : State α :=
  let v := written M source incoming gate gradient carry u
  let v := v.setReg "S" .ptr [64,16] (ptr "S" (fun e j => source e j-c.K*c.V))
  let v := v.setReg "DI" .ptr [64,16] (ptr "DI" (fun e j => source e j-c.K*c.V))
  let v := v.setReg "DS" .ptr [64,16] (ptr "DS" (fun e j => incoming e j-c.K*c.V))
  let v := v.setReg "DG" .ptr [] (fun _ => (("DG" : RegionName),gradient-c.KT*c.VT))
  v.setReg "D" .ptr [] (fun _ => (("D" : RegionName),gate-1))

theorem executes (M : Algebra α) (c : Shape) (source incoming : Fin 64 → Fin 16 → Nat)
    (gate gradient : Nat) (carry : Lane → α) (u : State α)
    (hs : u.regs .ptr [64,16] "S"=some (ptr "S" source))
    (hi : u.regs .ptr [64,16] "DI"=some (ptr "DI" source))
    (hx : u.regs .ptr [64,16] "DS"=some (ptr "DS" incoming))
    (hd : u.regs .ptr [] "D"=some (fun _ : TileIndex [] => (("D" : RegionName),gate)))
    (hg : u.regs .ptr [] "DG"=some (fun _ : TileIndex [] => (("DG" : RegionName),gradient)))
    (ha : u.regs .real [64,16] "Dacc"=some carry)
    (hk : u.regs .nat [] "NUM_K"=some (fun _ => c.KT))
    (hv : u.regs .nat [] "NUM_V"=some (fun _ => c.VT)) :
    run (engine M) (backwardBody c) u=some (result M c source incoming gate gradient carry u) := by
  simp (config := {maxSteps:=200000}) [backwardBody,loadScalar,loadMatrix,assign32,storeMatrix,
    storeScalar,retreat,matrixPtr,scalarPtr,matrix,scalar,KernelOps.update,stride,tileStride,
    KernelOps.gradient,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,bop,ofFloat,toFloat,Option.bind,
    ComputeDType.eraseDType,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
    State.setReg,FP.Structural.store,hs,hi,hx,hd,hg,ha,hk,hv,ptr,raw,scatter,
    result,written,pre,contraction,value,saved,loadedMatrix,loadedScalar,next,
    KernelReference.contraction,KernelReference.flattened,KernelReference.product,KernelReference.stored]
  all_goals try simp only [FP.Structural.bop,next,loadedMatrix,loadedScalar,KernelProgram.raw,
    KernelProgram.ptr,Broadcast.leftIndex,Broadcast.rightIndex,KernelStep.lane_eta]
  all_goals repeat' constructor
  all_goals rfl

#axiomsClean executes
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelReverseStep
