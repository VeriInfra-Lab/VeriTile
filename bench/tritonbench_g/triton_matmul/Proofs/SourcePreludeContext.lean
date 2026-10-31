import bench.tritonbench_g.triton_matmul.Proofs.SourceCoordinates
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1000000

theorem prelude_split (p : Parameters) (st : Storage) :
    prelude p st=setup p st++coordinates p st := by
  cases st <;> rfl

theorem preludeContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) (hm : 0<BM p) (hn : 0<BN p)
    (hf : SourceControl.first p s≤SourceControl.rows p) :
    Context p s (preludeExecution M D p st s).val ∧
      HasAcc p (fun _=>M.literal (some .fp32) .real 0) (preludeExecution M D p st s).val := by
  let u:=prefixExecution M D p st s
  have md:=SourceControl.prefix_metadata M D p st s
  have pm : u.val.regs .int [] "pid_m"=some (fun _=>(pidM p s : Int)) := by
    rw [SourceControl.prelude_tileM M D p st s hf,(SourceControl.coordinates_eq p s hm hn).2.2.1]
  have pn : u.val.regs .int [] "pid_n"=some (fun _=>(pidN p s : Int)) := by
    rw [SourceControl.prelude_tileN M D p st s hf,(SourceControl.coordinates_eq p s hm hn).2.2.2]
  let v:=coordinateExecution M D p st s u.val md pm pn
  have he : run (backend M D) (prelude p st) s=some v.val := by
    rw [prelude_split,run_append]
    simpa only [u.property,Option.bind_some] using v.property
  have eq : (preludeExecution M D p st s).val=v.val :=
    Option.some.inj ((preludeExecution M D p st s).property.symm.trans he)
  rw [eq]
  exact coordinateContext M D p st s u.val (SourceControl.prefix_memory M D p st s) md pm pn

#axiomsClean preludeContext
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
