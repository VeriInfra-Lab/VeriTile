import bench.tritonbench_g.rms_matmul_rbe.Proofs.Rotate
import bench.tritonbench_g.rms_matmul_rbe.Proofs.LoopFrame

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

def nativeRotation (plan : RotaryPlan) (p : Arguments) :=
  selectedRotary plan Bool.false Bool.false p

theorem native_rotation_source (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments) :
    selectedRotary plan fp8 outputFP8 p=nativeRotation plan p := by
  cases fp8 <;> cases outputFP8 <;> rfl

def nativeRotationExecution {α : Type} [Inhabited α] (a : Arithmetic α)
    (plan : RotaryPlan) (p : Arguments) (s : State α) (theta : α) (pm pn : Nat)
    (accumulator : Values α .real [16,64])
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    {next : State α // run a.model (nativeRotation plan p) s=some next} := by
  rcases plan with ⟨swap,fuse⟩
  cases swap <;> cases fuse
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [nativeRotation,selectedRotary,nativeRealNode,nativeImagNode,orderedFactors,
      fp,pairRef,pairMul,samePair,rotaryBody,epilogue,source,TypedSource.kernel,TypedSource.halfHalf,TypedSource.halfFP8,TypedSource.fp8Half,TypedSource.fp8FP8,
      ComputeKernel.surfaceBody,run,FP.Structural.step,evalExpr,evalComputeOp,
      evalOp_unfold,evalOffsets,evalDeltas,numeric,bop,natLt,numericLt,
      ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
      integralFloorDiv,integralMod,Arithmetic.model,ht,hm,hn,ha,
      Broadcast.leftIndex,Broadcast.rightIndex]
    rfl

def orderedMul (a : Arithmetic α) (swap : Bool) (left right : α) : α :=
  if swap then a.mul right left else a.mul left right

def orderedFma (a : Arithmetic α) (swap : Bool) (left right carry : α) : α :=
  if swap then a.fma right left carry else a.fma left right carry

def nativeRotatedPair (a : Arithmetic α) (plan : RotaryPlan) (p : Arguments)
    (theta : α) (pm pn : Nat) (accumulator : Values α .real [16,64])
    (row : Fin 16) (pair : Fin 32) : α × α :=
  let phase := pairAngle a p theta pm pn row pair
  let real := accumulator (row,evenLane pair,PUnit.unit)
  let imag := accumulator (row,oddLane pair,PUnit.unit)
  let cosine := a.cos phase
  let sine := a.sin phase
  (orderedFma a plan.swapFactors real cosine
      (a.neg (orderedMul a plan.swapFactors imag sine)),
    if plan.fuseRealSin then
      orderedFma a plan.swapFactors real sine (orderedMul a plan.swapFactors imag cosine)
    else orderedFma a plan.swapFactors imag cosine (orderedMul a plan.swapFactors real sine))

def nativeRotated (a : Arithmetic α) (plan : RotaryPlan) (p : Arguments)
    (theta : α) (pm pn : Nat) (accumulator : Values α .real [16,64]) :
    Values α .real [16,64] := fun i =>
  let pair : Fin 32 := ⟨i.2.1.val/2,by have := i.2.1.isLt; omega⟩
  let result := nativeRotatedPair a plan p theta pm pn accumulator i.1 pair
  if i.2.1.val%2=0 then result.1 else result.2

theorem native_rotation_frame {α : Type} [Inhabited α] (a : Arithmetic α)
    (plan : RotaryPlan) (p : Arguments) (s : State α) (theta : α) (pm pn : Nat)
    (accumulator : Values α .real [16,64])
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    (nativeRotationExecution a plan p s theta pm pn accumulator ht hm hn ha).val.mem=s.mem := by
  rcases plan with ⟨swap,fuse⟩
  cases swap <;> cases fuse <;> rfl

theorem native_rotation_header {α : Type} [Inhabited α] (a : Arithmetic α)
    (plan : RotaryPlan) (p : Arguments) (s : State α) (theta : α) (pm pn : Nat)
    (accumulator : Values α .real [16,64])
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    HeaderPreserved s
      (nativeRotationExecution a plan p s theta pm pn accumulator ht hm hn ha).val := by
  intro d sh name hn
  simp [headerNames] at hn
  rcases plan with ⟨swap,fuse⟩
  rcases hn with rfl|rfl|rfl|rfl|rfl
  all_goals cases swap <;> cases fuse <;> simp [nativeRotationExecution]

#axiomsClean native_rotation_source
#axiomsClean native_rotation_frame
#axiomsClean native_rotation_header
#specNonCircular nativeRotated avoiding [nativeRotationExecution,run,rotated]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
