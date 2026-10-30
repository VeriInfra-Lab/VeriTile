import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelOps

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelStep
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelOps
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

@[simp] theorem lane_eta (i : Lane) : (i.1,i.2.1,PUnit.unit)=i := by
  rcases i with ⟨e,j,⟨⟩⟩
  rfl

def loadedMatrix (M : Algebra α) (r : RegionName) (off : Fin 64 → Fin 16 → Nat) (u : State α) : Lane → α :=
  fun i => M.fp32Load (raw r .real off u i)
def loadedScalar (M : Algebra α) (r : RegionName) (off : Nat) (u : State α) : α :=
  M.fp32Load ((u.mem r off).read .real)
def forwardValue (M : Algebra α) (off : Fin 64 → Fin 16 → Nat) (gate : Nat)
    (carry : Lane → α) (u : State α) : Lane → α :=
  next M carry (loadedScalar M "D" gate u) (loadedMatrix M "S" off u)
def forwardPre (M : Algebra α) (off : Fin 64 → Fin 16 → Nat) (gate : Nat)
    (carry : Lane → α) (u : State α) : State α :=
  ((u.setReg "d_i" .real [] (fun _ => loadedScalar M "D" gate u)).setReg "S_i" .real [64,16]
    (loadedMatrix M "S" off u)).setReg "acc" .real [64,16] (forwardValue M off gate carry u)
def forwardResult (M : Algebra α) (c : Shape) (off target : Fin 64 → Fin 16 → Nat)
    (gate : Nat) (carry : Lane → α) (u : State α) : State α :=
  let value := forwardValue M off gate carry u
  let written := scatter "O" .real target (fun i => M.cast (some .fp32) .real .real (value i))
    (forwardPre M off gate carry u)
  ((written.setReg "D" .ptr [] (fun _ => (("D" : RegionName),gate+1))).setReg "S" .ptr [64,16]
    (ptr "S" (fun e j => off e j+c.K*c.V))).setReg "O" .ptr [64,16]
      (ptr "O" (fun e j => target e j+c.K*c.V))

theorem forward (M : Algebra α) (c : Shape) (off target : Fin 64 → Fin 16 → Nat)
    (gate : Nat) (carry : Lane → α) (u : State α)
    (hs : u.regs .ptr [64,16] "S"=some (ptr "S" off))
    (ho : u.regs .ptr [64,16] "O"=some (ptr "O" target))
    (hd : u.regs .ptr [] "D"=some (fun _ : TileIndex [] => (("D" : RegionName),gate)))
    (ha : u.regs .real [64,16] "acc"=some carry) :
    run (engine M) (forwardBody c) u=some (forwardResult M c off target gate carry u) := by
  simp (config := {maxSteps:=150000}) [forwardBody,loadScalar,loadMatrix,assign32,storeMatrix,advance,
    matrixPtr,scalarPtr,matrix,scalar,update,stride,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,bop,ofFloat,toFloat,Option.bind,
    ComputeDType.eraseDType,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
    State.setReg,FP.Structural.store,hs,ho,hd,ha,ptr,raw,scatter,
    forwardResult,forwardPre,forwardValue,loadedMatrix,loadedScalar,next]
  all_goals try simp only [FP.Structural.bop,next,loadedMatrix,loadedScalar,KernelProgram.raw,
    KernelProgram.ptr,Broadcast.leftIndex,Broadcast.rightIndex,lane_eta]
  all_goals repeat' constructor
  all_goals rfl

#axiomsClean forward
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelStep
