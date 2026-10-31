import bench.tritonbench_g.triton_matmul.Proofs.SourcePrelude
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 700000
set_option linter.unusedSimpArgs false

def Metadata (p : Parameters) {α : Type} (s : State α) : Prop :=
  s.regs .int [] "M"=some (fun _=>(p.M : Int)) ∧
  s.regs .int [] "N"=some (fun _=>(p.N : Int)) ∧
  s.regs .int [] "K"=some (fun _=>(p.K : Int)) ∧
  s.regs .int [] "stride_am"=some (fun _=>(p.stride_am : Int)) ∧
  s.regs .int [] "stride_ak"=some (fun _=>(p.stride_ak : Int)) ∧
  s.regs .int [] "stride_bk"=some (fun _=>(p.stride_bk : Int)) ∧
  s.regs .int [] "stride_bn"=some (fun _=>(p.stride_bn : Int)) ∧
  s.regs .int [] "stride_cm"=some (fun _=>(p.stride_cm : Int)) ∧
  s.regs .int [] "stride_cn"=some (fun _=>(p.stride_cn : Int))

theorem prelude_memory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    (preludeExecution M D p st s).val.mem=s.mem := by
  cases st <;> simp [preludeExecution]

theorem prelude_pids {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    (preludeExecution M D p st s).val.pids=s.pids := by
  cases st <;> simp [preludeExecution]

theorem prelude_metadata {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    Metadata p (preludeExecution M D p st s).val := by
  cases st <;> simp [Metadata,preludeExecution]

#axiomsClean prelude_memory
#axiomsClean prelude_pids
#axiomsClean prelude_metadata
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
