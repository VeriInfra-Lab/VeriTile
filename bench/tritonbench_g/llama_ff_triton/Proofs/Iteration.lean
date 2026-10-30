import bench.tritonbench_g.llama_ff_triton.Proofs.Reference

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def loopBody (native : Bool) (p : Parameters) := if native then compiledBody p else body p

theorem mask_eq (p : Parameters) (step lane : Nat) :
    ((lane : Int) < (p.inner : Int) - (step : Int)*64) ↔ activeK p step lane := by
  unfold activeK
  omega

theorem initialInvariant {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (s : State α) :
    Invariant a native p s 0 (prefixExecution a.model p s).val := by
  refine ⟨prefixIndices a.model p s, ?_, ?_, ?_⟩ <;>
    simp [prefixExecution, accumulation, moment, Arithmetic.model]

def bodyExecution {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial current : State α) (k : Nat)
    (h : Invariant a native p initial k current) (bytes : ByteValues initial) :
    {t : State α // run a.model (loopBody native p)
      (current.setReg "step" .int [] (fun _ => (k : Int))) = some t} := by
  have safe (weight : RegionName) (hw : weight ∈ (["RMS","W1","W3"] : List RegionName))
      (offset : Nat) (test : Prop) [Decidable test] :
      Float8E5Bits.isInt8 (if test then (initial.mem weight offset).read .int else 0)=Bool.true := by
    split_ifs
    · exact bytes weight hw offset
    · decide
  cases native <;> cases hp : p.byte
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp [loopBody, compiledBody, dot1Index, dot3Index, carryNode, directWeightNode,
      body, source, Source.kernel, Source.half, Source.byte, hp,
      ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
      evalOp_unfold, evalOffsets, evalDeltas, numeric, bop, natLt, numericLt,
      ComputeDType.eraseDType, ofFloat, toFloat, Region.cast, Option.bind,
      integralFloorDiv, integralMod, Arithmetic.model,
      h.memory, h.k, h.sak, h.w1k, h.w3k, h.rms, h.offk,
      h.aptrs, h.bptrs, h.cptrs, h.rptrs, h.acc1, h.acc2, h.sum,
      Broadcast.leftIndex, Broadcast.rightIndex, mask_eq,
      Int32Bits.valuesValid, Int32Bits.inRange,
      safe "RMS" (by simp), safe "W1" (by simp), safe "W3" (by simp)]
    rfl

end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
