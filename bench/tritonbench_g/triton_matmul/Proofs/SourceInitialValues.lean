import bench.tritonbench_g.triton_matmul.Proofs.SourceMetadata
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 700000
set_option linter.unusedSimpArgs false

theorem prelude_offs_k {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    (preludeExecution M D p st s).val.regs .int [BK p] "offs_k"=
      some (fun i=>(i.1.val : Int)) := by
  cases st <;> simp [preludeExecution,BK]

theorem prelude_accumulator {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    (preludeExecution M D p st s).val.regs .real [BM p,BN p] "accumulator"=
      some (fun _=>M.literal (some .fp32) .real 0) := by
  cases st <;> simp [preludeExecution,backend]

#axiomsClean prelude_offs_k
#axiomsClean prelude_accumulator
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
