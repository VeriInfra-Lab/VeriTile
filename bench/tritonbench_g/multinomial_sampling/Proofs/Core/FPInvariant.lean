import bench.tritonbench_g.multinomial_sampling.Proofs.Core.FPReference
namespace VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
structure Invariant (P : Primitives α) (p : Parameters) (initial : State α)
    (st : Accumulator α p) (current : State α) : Prop where
  memory : current.mem=initial.mem
  off : current.regs .nat [p.block] "off"=some (row p initial)
  n_off : current.regs .nat [p.blockTokens] "n_off"=some (fun i => i.1.val)
  mask : current.regs .bool [p.block] "off_mask"=some (fun i => decide (active p initial i))
  samp : current.regs .real [p.block,1] "samp"=some (fun i => sample P p initial (i.1,PUnit.unit))
  acc : current.regs .real [p.block] "acc"=some st.acc
  output : current.regs .int [p.block] "output"=some st.output
end
end VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
