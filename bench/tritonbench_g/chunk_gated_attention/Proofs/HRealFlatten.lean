import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealSafety
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
open RealHMath RealHProgram VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Bounds
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 600000
set_option linter.unusedSimpArgs false

private theorem gate_operations_flatten (m t n : Nat) :
    (((Op.expandDim ⟨1,by simp⟩ (.exp (.ref .real [m] "b_gn")) : Op .real [m,1]).FlattenOk ∧
      (Op.expandDim ⟨1,by simp⟩ (.ref .real [m] "b_gn") : Op .real [m,1]).FlattenOk ∧
      (Op.dot (batch:=[]) (.ref .real [m,t] "b_k") (.ref .real [t,n] "b_v")
        {allowTF32:=some Bool.false}).FlattenOk) ∧
     (Op.expandDim ⟨0,by simp⟩ (.exp (.ref .real [n] "b_gn")) : Op .real [1,n]).FlattenOk ∧
     (Op.expandDim ⟨0,by simp⟩ (.ref .real [n] "b_gn") : Op .real [1,n]).FlattenOk ∧
     (Op.dot (batch:=[]) (.ref .real [m,t] "b_k") (.ref .real [t,n] "b_v")
       {allowTF32:=some Bool.false}).FlattenOk) := by
  repeat' apply And.intro
  all_goals simp [Op.FlattenOk.eq_def]

theorem source_flatten_ok (p : Parameters) (r : RealHProgram.Regions)
    (dot gate history seed final : FloatDType) :
    (kernel p r dot gate history seed final).eraseDType.toAlgKernel.FlattenOk := by
  rw [source_connection,Kernel.FlattenOk,decomposition]
  simp only [prelude_ast,seed_ast,iteration_structure,iteration_decomposition,
    iteration_prefix_ast,gate_key_ast,gate_value_ast,final_ast]
  simp only [RealHProgram.iterationPrelude,RealHProgram.gateKey,RealHProgram.gateValue,
    List.drop,List.cons_append,List.nil_append,List.append_assoc]
  cases hp:p.swapped
  all_goals simp only [updateStmt,hp,Bool.false_eq_true,↓reduceIte,storeCarry,carry,
    RealHProgram.dot,keyPtr,valuePtr,historyPtr,statePtr,endpointPtr,integer,reg,natReg,add,mul,load1,load2]
  all_goals simp only [List.cons_append,List.nil_append,List.drop,StmtList.FlattenOk,Stmt.FlattenOk]
  all_goals simp only [Op.flattenOk_truncDiv,Op.flattenOk_truncMod,Op.FlattenOk,
    and_true,true_and,implies_true,List.mem_cons,List.mem_singleton,List.not_mem_nil,
    or_imp,forall_and,forall_eq,forall_const]
  all_goals simp
  all_goals exact gate_operations_flatten p.BK p.BT p.BV

#axiomsClean source_flatten_ok
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
