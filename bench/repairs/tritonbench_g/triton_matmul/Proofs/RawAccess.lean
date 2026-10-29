import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawWritten

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section

theorem addressA_model {α : Type} (p : Parameters) (s : State α)
    (hm : (control p s).pidM = (pidM p s : Int)) (k : Nat) (i : AIndex p)
    (hi : innerMask p k i.2.1 = Bool.true) :
    addressA p s k i = aOffset p s k i := by
  simp only [addressA,offsetM_model p s hm,innerOffset,hi,ite_true,
    ← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast,aOffset]
  ring

theorem addressB_model {α : Type} (p : Parameters) (s : State α)
    (hn : (control p s).pidN = (pidN p s : Int)) (k : Nat) (i : BIndex p)
    (hi : innerMask p k i.1 = Bool.true) :
    addressB p s k i = bOffset p s k i := by
  simp only [addressB,offsetN_model p s hn,innerOffset,hi,ite_true,
    ← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast,bOffset]
  ring

#axiomsClean addressA_model
#axiomsClean addressB_model
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
