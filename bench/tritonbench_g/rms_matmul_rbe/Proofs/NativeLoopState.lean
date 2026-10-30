import bench.tritonbench_g.rms_matmul_rbe.Proofs.NativeBody
import bench.tritonbench_g.rms_matmul_rbe.Proofs.LoopFrame

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

/-- Selected-native recurrence: the K8 instruction stream consumes the live
accumulator. All source loads, casts, powi and pointer updates are retained. -/
def nativeNextValues {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s : State α) (v : LoopValues α) (t : Nat) : LoopValues α where
  keys := v.keys
  xptrs := fun i => ((v.xptrs i).1,(v.xptrs i).2+64*p.sxk)
  wptrs := fun i => ((v.wptrs i).1,(v.wptrs i).2+64*p.swk)
  rmsptrs := fun i => ((v.rmsptrs i).1,(v.rmsptrs i).2+64*p.srms)
  moments := fun i => a.add (v.moments i) (a.powi (xTile a p s v t i) 2)
  accumulator := tensor a (batch:=[]) (scaledTile a fp8 p s v t)
    (weightTile a fp8 p s v t) v.accumulator

theorem native_body_values {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : Int8Memory s) (t : Nat) :
    LoopRegisters (nativeNextValues a fp8 p s v t)
      (nativeBodyExecution a fp8 outputFP8 p s v h bytes t).val := by
  have zeroScientific : (0.0:ℝ)=0 := by norm_num
  cases fp8 <;> cases outputFP8
  all_goals
    constructor <;>
      simp [nativeBodyExecution,nativeNextValues,xTile,rmsTile,weightTile,scaledTile,liveKey,
        kernelCast,binary,FP.Structural.bop,h.keys,ofFloat,toFloat,
        Broadcast.leftIndex,Broadcast.rightIndex,Fin.eq_zero,zeroScientific]
    all_goals try rfl

theorem native_body_frame {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : Int8Memory s) (t : Nat) :
    (nativeBodyExecution a fp8 outputFP8 p s v h bytes t).val.mem=s.mem := by
  cases fp8 <;> cases outputFP8 <;> rfl

theorem native_body_header {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : Int8Memory s) (t : Nat) :
    HeaderPreserved s (nativeBodyExecution a fp8 outputFP8 p s v h bytes t).val := by
  intro d sh name hn
  simp [headerNames] at hn
  rcases hn with rfl|rfl|rfl|rfl|rfl
  all_goals cases fp8 <;> cases outputFP8 <;> simp [nativeBodyExecution]

#axiomsClean native_body_values
#axiomsClean native_body_frame
#axiomsClean native_body_header
#specNonCircular nativeNextValues avoiding [nativeBodyExecution,run]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
