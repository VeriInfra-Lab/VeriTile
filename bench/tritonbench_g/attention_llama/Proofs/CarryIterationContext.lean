import bench.tritonbench_g.attention_llama.Proofs.CarryIteration
import bench.tritonbench_g.attention_llama.Proofs.ReductionValues

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarrySource
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open Source
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem iteration_context_of_reached (c : Config) (s t : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) (h : Context c s scale n a t) (bytes : ReachedBytes c s) :
    Context c s scale (n+1) (advance c s scale n a)
      (iteration_of_reached c s t scale n a h bytes).val := by
  rcases c with ⟨Q,K,V,Out,sqz,sqh,sqm,sqk,skz,skh,skn,skk,svz,svh,svk,svn,soz,soh,som,son,NH,H,N,SP,D,causal,fp8⟩
  cases fp8 <;> cases causal
  all_goals constructor <;> simp [iteration_of_reached,advance,score_function,row_reduction,scores,loaded,
    h.scalar,h.maximum,h.denominator,h.accumulator,h.query,h.rows,h.columns,
    h.batch,h.head,h.key,h.value,h.memory,h.pids,
    dotValues,transposeValues,reductionInputs,bop,ofFloat,toFloat,
    Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    TileShape.insertAxisIndex,TileShape.replaceAxisIndex,
    TileShape.indexToList,TileShape.axisDim]
  all_goals funext i
  all_goals simp [score_function,row_reduction,scores,loaded,pointers,base,rows,Nat.add_mul,Nat.mul_add,Nat.add_assoc,
    dotValues,transposeValues,reductionInputs,bop,ofFloat,toFloat,
    Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    TileShape.insertAxisIndex,TileShape.replaceAxisIndex,TileShape.indexToList,TileShape.axisDim]
  all_goals norm_num
  all_goals
    repeat' first
      | rfl
      | apply And.intro
      | apply funext; intro j

theorem iteration_context (c : Config) (s t : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) (h : Context c s scale n a t) (bytes : ByteRange s) :
    Context c s scale (n+1) (advance c s scale n a)
      (iteration c s t scale n a h bytes).val :=
  iteration_context_of_reached c s t scale n a h (bytes.reached c s)

#axiomsClean iteration_context_of_reached
#axiomsClean iteration_context
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarrySource
