import bench.tritonbench_g.rms_matmul_rbe.Proofs.BodyExecution

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def liveKey (p : Arguments) (v : LoopValues α) (t : Nat) (k : Fin 64) : Prop :=
  v.keys (k,PUnit.unit)<p.K-t*64
instance (p : Arguments) (v : LoopValues α) (t : Nat) (k : Fin 64) :
    Decidable (liveKey p v t k) := inferInstanceAs (Decidable (_<_))

def xTile {α : Type} [Inhabited α] (a : Arithmetic α) (p : Arguments)
    (s : State α) (v : LoopValues α) (t : Nat) : Values α .real [16,64] := fun i =>
  if liveKey p v t i.2.1 then (s.mem (v.xptrs i).1 (v.xptrs i).2).read .real
  else a.literal .real 0

def rmsTile {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool) (p : Arguments)
    (s : State α) (v : LoopValues α) (t : Nat) (k : Fin 64) : α :=
  let ptr := v.rmsptrs (⟨0,by decide⟩,k,PUnit.unit)
  let half := if fp8 then
    let byte := if liveKey p v t k then (s.mem ptr.1 ptr.2).read .int else 0
    kernelCast a .f8e5 .fp16 (a.e5Bits .f8e5 (Float8E5Bits.ofInt byte))
  else if liveKey p v t k then (s.mem ptr.1 ptr.2).read .fp16
    else kernelCast a .real .fp16 (a.literal .real 0)
  kernelCast a .fp16 .real half

def weightTile {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool) (p : Arguments)
    (s : State α) (v : LoopValues α) (t : Nat) : Values α .real [64,64] := fun i =>
  let ptr := v.wptrs i
  let half := if fp8 then
    let byte := if liveKey p v t i.1 then (s.mem ptr.1 ptr.2).read .int else 0
    kernelCast a .real .fp16 (kernelCast a .f8e5 .real
      (a.e5Bits .f8e5 (Float8E5Bits.ofInt byte)))
  else if liveKey p v t i.1 then (s.mem ptr.1 ptr.2).read .fp16
    else kernelCast a .real .fp16 (a.literal .real 0)
  kernelCast a .fp16 .real half

def scaledTile {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool) (p : Arguments)
    (s : State α) (v : LoopValues α) (t : Nat) : Values α .real [16,64] := fun i =>
  a.mul (xTile a p s v t i) (rmsTile a fp8 p s v t i.2.1)

/-- Explicit scalar recurrence for one source K64 iteration. It reads only
the pre-iteration memory and six loop values, and never calls the evaluator. -/
def nextValues {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool) (p : Arguments)
    (s : State α) (v : LoopValues α) (t : Nat) : LoopValues α where
  keys := v.keys
  xptrs := fun i => ((v.xptrs i).1,(v.xptrs i).2+64*p.sxk)
  wptrs := fun i => ((v.wptrs i).1,(v.wptrs i).2+64*p.swk)
  rmsptrs := fun i => ((v.rmsptrs i).1,(v.rmsptrs i).2+64*p.srms)
  moments := fun i => a.add (v.moments i) (a.powi (xTile a p s v t i) 2)
  accumulator := fun i => a.add (v.accumulator i)
    (tensor a (batch:=[]) (scaledTile a fp8 p s v t) (weightTile a fp8 p s v t) (fun _ => a.zero) i)

theorem body_values {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : Int8Memory s) (t : Nat) :
    LoopRegisters (nextValues a fp8 p s v t)
      (bodyExecution a fp8 outputFP8 p s v h bytes t).val := by
  have zeroScientific : (0.0:ℝ)=0 := by norm_num
  cases fp8 <;> cases outputFP8
  all_goals
    constructor <;>
      simp [bodyExecution,nextValues,xTile,rmsTile,weightTile,scaledTile,liveKey,
        kernelCast,binary,FP.Structural.bop,h.keys,ofFloat,toFloat,
        Broadcast.leftIndex,Broadcast.rightIndex,Fin.eq_zero,zeroScientific]
    all_goals try rfl

theorem body_frame {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : Int8Memory s) (t : Nat) :
    (bodyExecution a fp8 outputFP8 p s v h bytes t).val.mem=s.mem := by
  cases fp8 <;> cases outputFP8 <;> rfl

#axiomsClean body_values
#axiomsClean body_frame
#specNonCircular nextValues avoiding [bodyExecution,run]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
