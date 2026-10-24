import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReverseStep
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReversePointers

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelPost
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelOps KernelStep
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def pre (M : Algebra α) (c : Shape) (p : Program c) (carry : Lane → α) (u : State α) : State α :=
  let v := u.setReg "DL" .ptr [64,16] (ptr "DL" (fun e j => dseedAddress c p e.val j.val))
  let v := v.setReg "DS_i" .real [64,16]
    (loadedMatrix M "DS" (fun e j => address c p 0 e.val j.val) u)
  let v := v.setReg "d_i" .real [] (fun _ => loadedScalar M "D" (gateAddress c p 0) u)
  v.setReg "Dacc" .real [64,16]
    (KernelReverseStep.value M (fun e j => address c p 0 e.val j.val) (gateAddress c p 0) carry u)
def result (M : Algebra α) (c : Shape) (p : Program c) (carry : Lane → α) (u : State α) : State α :=
  scatter "DL" .fp32 (fun e j => dseedAddress c p e.val j.val)
    (fun i => M.cast (some .fp32) .real .fp32
      (KernelReverseStep.value M (fun e j => address c p 0 e.val j.val) (gateAddress c p 0) carry u i))
    (pre M c p carry u)

theorem executes (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c)
    (carry : Lane → α) (u : State α) (fixed : KernelReversePointers.Fixed c p u)
    (hx : u.regs .ptr [64,16] "DS"=some (ptr "DS" (fun e j => address c p 0 e.val j.val)))
    (hd : u.regs .ptr [] "D"=some (fun _ : TileIndex [] => (("D" : RegionName),gateAddress c p 0)))
    (ha : u.regs .real [64,16] "Dacc"=some carry) :
    run (engine M) (after .backward c s seed) u=some (result M c p carry u) := by
  simp only [after,parts,source,Kernels.backward,
    TritonBenchG.ChunkGateRecurrence.TypedSource.backward,
    TritonBenchG.ChunkGateRecurrence.TypedSource.specializeBackward,
    TritonBenchG.ChunkGateRecurrence.TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,
    String.reduceEq,beq_self_eq_true,↓reduceIte]
  simp (config := {maxSteps:=200000}) [run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,bop,ofFloat,toFloat,Option.bind,
    ComputeDType.eraseDType,
    State.setReg,FP.Structural.store,hx,hd,ha,fixed.head,fixed.key,fixed.value,fixed.offsK,fixed.offsV,
    ptr,raw,scatter,result,pre,KernelReverseStep.value,loadedMatrix,loadedScalar,next,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,dseedAddress]
  all_goals try simp only [FP.Structural.bop,next,loadedMatrix,loadedScalar,KernelProgram.raw,
    KernelProgram.ptr,Broadcast.leftIndex,Broadcast.rightIndex,KernelStep.lane_eta]
  apply congrArg (fun z : State α => (TileShape.allIndices [64,16]).foldl
    (fun v i => v.write "DL" (dseedAddress c p i.1.val i.2.1.val)
      (.mk .fp32 (M.cast (some .fp32) .real .fp32
        (KernelReverseStep.value M (fun e j => address c p 0 e.val j.val)
          (gateAddress c p 0) carry u i)))) z)
  congr 1
  funext d sh name
  split_ifs <;> subst_vars
  all_goals simp [FP.Structural.bop,next,loadedMatrix,loadedScalar,KernelProgram.raw,
    KernelProgram.ptr,KernelReverseStep.value,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.dropInsertedIndex,KernelStep.lane_eta,KernelProgram.dseedAddress]
  all_goals funext i
  all_goals simp [FP.Structural.bop,KernelOps.next,KernelStep.loadedMatrix,KernelStep.loadedScalar,
    KernelProgram.raw,KernelProgram.ptr,KernelReverseStep.value,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.dropInsertedIndex,KernelStep.lane_eta,KernelProgram.dseedAddress]

  all_goals rfl

#axiomsClean executes
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelPost
