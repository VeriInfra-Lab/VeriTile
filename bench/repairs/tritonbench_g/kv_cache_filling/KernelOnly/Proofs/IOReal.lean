import bench.repairs.tritonbench_g.kv_cache_filling.KernelOnly.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.KvCacheFilling.Interface
open VeriTile.Triton FP.Structural
noncomputable section

theorem real_implements (r : Request) (d : Domain r) (j : Job) (version : Bool) (pid : Nat) :
    (programIO QuantReal.ops r j version pid).ImplementsOn (inputDomain r j)
      (fun vals => result r (SharedTarget.idealValue r (keyRow r vals) (valueRow r vals))) := by
  intro vals dom bounds hb s hin
  obtain ⟨safe,f,run,out,frame⟩ := implements QuantReal.ops r d j version pid vals dom bounds hb s hin
  refine ⟨safe,f,run,?_,frame⟩
  have eq : target QuantReal.ops r vals =
      result r (SharedTarget.idealValue r (keyRow r vals) (valueRow r vals)) :=
    congrArg (result r) (funext (SharedTarget.real_value_independent r d _ _))
  simpa only [eq] using out

#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.KvCacheFilling.Interface
