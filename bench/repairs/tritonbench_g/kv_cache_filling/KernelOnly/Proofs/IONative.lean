import bench.repairs.tritonbench_g.kv_cache_filling.KernelOnly.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.KvCacheFilling.Interface
open VeriTile.Triton FP.Structural
noncomputable section

/-- Both sources implement the same independently defined ordered native row
formula, so their complete active typed outputs agree. -/
theorem equivalent {α : Type} [Inhabited α] (a : QuantProfile.Ops α) (r : Request) (d : Domain r)
    (j : Job) (pid : Nat) :
    (programIO a r j Bool.false pid).EquivalentOn (candidate a r j pid) (inputDomain r j) := by
  exact ProgramIO.EquivalentOn.of_implementsOn
    (implements a r d j Bool.false pid) (implements a r d j Bool.true pid)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.KvCacheFilling.Interface
