import bench.tritonbench_g.llama_ff_triton.Proofs.GateSource

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.NativeStages
open VeriTile.Triton FP.Structural Execution
noncomputable section

/-- Exact syntax certificate for the original ordered activation site. The
existing native gate consumes the pre-multiplication acc1/acc2 and `a_norm`;
its FMA and multiplication order remain the explicitly selected compiler
boundary, with no claimed floating identity between the two expressions. -/
theorem original_gate_operands (p : Parameters) :
    ((epilogue p).drop 2).take 3 = [
      .assign .real [16,16] "acc1" (.compute (.alg .fp32
        (.mul .real (.consSame (.consR .nil)) (.ref .real [16,16] "acc1")
          (.expandDim (shape := [16]) ⟨1,by decide⟩ (.ref .real [16] "a_norm"))))),
      .assign .real [16,16] "acc2" (.compute (.alg .fp32
        (.mul .real (.consSame (.consR .nil)) (.ref .real [16,16] "acc2")
          (.expandDim (shape := [16]) ⟨1,by decide⟩ (.ref .real [16] "a_norm"))))),
      .assign .real [16,16] "accumulator" (.compute (.alg .fp32
        (.mul .real (.consSame (.consSame .nil))
          (.mul .real (.consSame (.consSame .nil)) (.ref .real [16,16] "acc1")
            (.sigmoid (.ref .real [16,16] "acc1"))) (.ref .real [16,16] "acc2")))) ] := by
  rcases p with ⟨M,N,K,b,e⟩
  cases b <;> rfl

#axiomsClean original_gate_operands
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.NativeStages
