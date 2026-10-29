import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Execution

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false


theorem preludeContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Problem) (b : Buffers) (s : State α) :
    Context p s (preludeExecution M D p b s).val ∧
      HasAcc (fun _ => M.literal (some .fp32) .real 0) (preludeExecution M D p b s).val := by
  by_cases hm : p.M%128 = 0 <;> by_cases hn : p.N%256 = 0 <;> by_cases hk : p.K%64 = 0
  all_goals constructor
  all_goals try constructor
  all_goals simp [preludeExecution,hm,hn,hk,backend,rows,columns,trips,group,perGroup,firstM,
    groupSize,pidM,pidN,row,col,qDiv,qMod,min_form,HasAcc,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try rfl
  all_goals funext i
  all_goals try simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
