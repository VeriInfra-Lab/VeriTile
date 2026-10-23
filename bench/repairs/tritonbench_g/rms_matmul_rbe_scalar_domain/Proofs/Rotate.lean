import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Arithmetic
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Lowering

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

def rotation (p : Arguments) := rotaryBody Bool.false Bool.false p

theorem rotation_source (fp8 outputFP8 : Bool) (p : Arguments) :
    rotaryBody fp8 outputFP8 p=rotation p := by
  cases fp8 <;> cases outputFP8 <;> rfl

/-- The true branch of the actual source rotary epilogue executes for every
opaque floating carrier, including nonfinite values; no numeric laws are
needed to establish success of reshape/split/join or transcendental calls. -/
def rotationExecution {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Arguments) (s : State α) (theta : α) (pm pn : Nat)
    (accumulator : Values α .real [16,64])
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    {next : State α // run a.model (rotation p) s=some next} := by
  refine ⟨?_,?_⟩
  swap
  simp [rotation,rotaryBody,epilogue,source,TypedSource.kernel,ComputeKernel.surfaceBody,run,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
    numeric,bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
    Region.cast,Option.bind,integralFloorDiv,integralMod,Arithmetic.model,ht,hm,hn,ha,
    Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

def pairColumn (pn : Nat) (pair : Fin 32) : Nat := pn*64+2*pair.val
def pairAngle (a : Arithmetic α) (p : Arguments) (theta : α) (pm pn : Nat)
    (row : Fin 16) (pair : Fin 32) : α :=
  a.divide (a.ofInt (p.start+((pm*16+row.val:Nat):Int)))
    (a.pow theta (a.divide (a.ofInt (((pairColumn pn pair)%p.headDim:Nat):Int))
      (a.literalNat p.headDim)))

def evenLane (pair : Fin 32) : Fin 64 := ⟨2*pair.val,by have := pair.isLt; omega⟩
def oddLane (pair : Fin 32) : Fin 64 := ⟨2*pair.val+1,by have := pair.isLt; omega⟩

def rotatedPair (a : Arithmetic α) (p : Arguments) (theta : α) (pm pn : Nat)
    (accumulator : Values α .real [16,64]) (row : Fin 16) (pair : Fin 32) : α × α :=
  let phase := pairAngle a p theta pm pn row pair
  let real := accumulator (row,evenLane pair,PUnit.unit)
  let imag := accumulator (row,oddLane pair,PUnit.unit)
  (a.sub (a.mul real (a.cos phase)) (a.mul imag (a.sin phase)),
   a.add (a.mul real (a.sin phase)) (a.mul imag (a.cos phase)))

def rotated (a : Arithmetic α) (p : Arguments) (theta : α) (pm pn : Nat)
    (accumulator : Values α .real [16,64]) : Values α .real [16,64] := fun i =>
  let pair : Fin 32 := ⟨i.2.1.val/2,by have := i.2.1.isLt; omega⟩
  let result := rotatedPair a p theta pm pn accumulator i.1 pair
  if i.2.1.val%2=0 then result.1 else result.2

theorem rotation_frame {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Arguments) (s : State α) (theta : α) (pm pn : Nat)
    (accumulator : Values α .real [16,64])
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    (rotationExecution a p s theta pm pn accumulator ht hm hn ha).val.mem=s.mem := rfl

#axiomsClean rotation_source
#axiomsClean rotation_frame
#specNonCircular rotated avoiding [rotationExecution,run]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
