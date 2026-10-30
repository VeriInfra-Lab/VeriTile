import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.CandidateProgram
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.FPValues

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.CandidateValues
open VeriTile.Triton FP.Structural FPValues CandidateProgram
noncomputable section
variable {α : Type} [Inhabited α]

def updated (M : Algebra α) (c : Config) (input : Values α .real [c.BT,BS])
    (acc : Values α .real [BS]) : Values α .real [BS] :=
  fun j => M.binary (some .fp32) .real .add
    (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false input j) (acc j)

theorem carry_eval (M : Algebra α) (dot : DotBackend α) (c : Config)
    (input : Values α .real [c.BT,BS]) (acc : Values α .real [BS]) (u : State α)
    (ha : u.regs .real [BS] "b_z"=some acc)
    (hb : u.regs .real [c.BT,BS] "b_s"=some input) :
    FP.Structural.evalOp (machine M dot) (some .fp32) (sumLeftCarryOp c) u=some (updated M c input acc) := by
  simp only [sumLeftCarryOp,FP.Structural.evalOp,ha,hb]
  rfl

#axiomsClean carry_eval
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.CandidateValues
