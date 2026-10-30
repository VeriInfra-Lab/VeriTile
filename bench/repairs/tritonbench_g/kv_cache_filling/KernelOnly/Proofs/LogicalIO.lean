import bench.repairs.tritonbench_g.kv_cache_filling.KernelOnly.Proofs.IOReal

namespace VeriTile.Bench.Repairs.KvCacheFilling.LogicalIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Full logical judgment, including the unchanged domain, source safety,
successful execution, masked observations and exterior frame. -/
theorem judgment_iff (r : Request) (j : Job) (version : Bool) (pid : Nat)
    (f : Inputs r → Outputs r) :
    (io QuantReal.ops r j version pid).ImplementsOn (domain r j) f ↔
      (Interface.programIO QuantReal.ops r j version pid).ImplementsOn (Interface.inputDomain r j)
        (fun raw => (outputs r).symm (f (inputs r raw))) := Iff.rfl

theorem real_implements (r : Request) (d : Domain r) (j : Job) (version : Bool) (pid : Nat) :
    (io QuantReal.ops r j version pid).ImplementsOn (domain r j)
      (fun x => outputs r (Interface.result r (SharedTarget.idealValue r
        (extend x.key) (extend x.value)))) := by
  rw [judgment_iff]
  simpa only [Equiv.symm_apply_apply] using Interface.real_implements r d j version pid

#axiomsClean judgment_iff
#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.KvCacheFilling.LogicalIO
