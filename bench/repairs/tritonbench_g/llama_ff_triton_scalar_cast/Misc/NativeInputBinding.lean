import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.InputBinding
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Misc.NativeReference

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface
open VeriTile.Triton FP.Structural Execution
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem Reads.native_target {xs : Inputs α} {s : State α} (h : Reads xs s)
    (a : Arithmetic α) (g : GateInstructions α) (p : Parameters) (row column : Nat) :
    NativeReference.target a g p s row column = Reference.nativeValue a g p xs row column := by
  have hx (r t : Nat) : rowX a p s r t = Reference.xValue a p xs r t := by
    funext k
    simp [rowX,Reference.xValue,h.input]
  have hr (t : Nat) : (fun k : Fin 64 => Execution.rmsValue a p s t k.val) = Reference.rmsValue a p xs t := by
    funext k
    simp [Execution.rmsValue,Execution.rmsByte,activeK,Reference.rmsValue,h.rmsByte,h.rmsHalf]
  have hs (r t : Nat) : rowScaled a p s r t = Reference.scaled a p xs r t := by
    funext k
    simpa only [rowScaled,Reference.scaled,hx] using
      congrArg (a.binary none .fp16 .mul (Reference.xValue a p xs r t k)) (congrFun (hr t) k)
  have hw1 (c t : Nat) : nativeRowW a p s "W1" c t = Reference.weightValue a p xs Bool.false c t := by
    funext k
    simp [nativeRowW,Reference.weightValue,h.w1Byte,h.w1Half]
  have hw3 (c t : Nat) : nativeRowW a p s "W3" c t = Reference.weightValue a p xs Bool.true c t := by
    funext k
    simp [nativeRowW,Reference.weightValue,h.w3Byte,h.w3Half]
  have hm (r : Nat) : NativeReference.moment a p s r (steps p) = Reference.moment a p xs r := by
    funext k
    simp only [NativeReference.moment,Reference.moment,hx]
  simp only [NativeReference.target,NativeReference.normalization,NativeReference.accumulation,
    Reference.nativeValue,Reference.normalization,Reference.accumulation,hs,hw1,hw3,hm]

#axiomsClean Reads.native_target
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface
