import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealPointers
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1400000
set_option linter.unusedSimpArgs false

def iterationPrefix (p : Parameters) (r : RealHProgram.Regions) : List Stmt :=
  (iteration p r).take 6

def gateKey (p : Parameters) (r : RealHProgram.Regions) : List Stmt :=
  match ((iteration p r)[6]?).getD (.ifThen (.constBool Bool.false) []) with
  | .ifThenElse _ yes _=>yes.take 6
  | _=>[]

def gateValue (p : Parameters) (r : RealHProgram.Regions) : List Stmt :=
  match ((iteration p r)[6]?).getD (.ifThen (.constBool Bool.false) []) with
  | .ifThenElse _ _ no=>no.take 6
  | _=>[]

local macro "reduce_iteration" h:term : tactic => `(tactic|
  (dsimp only [iterationPrefix,gateKey,gateValue,iteration,loopStatement,algorithm,projection,Kernel.eraseDType]
   simp only [($h),Bool.false_eq_true,↓reduceDIte,↓reduceIte]
   simp only [ScalarArgument.integer,ScalarArgument.statement,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType_truncDiv,Op.eraseDType_truncMod,Op.eraseDType_castFloat,Op.eraseDType,
    MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]
   simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]
   rfl))

theorem iteration_decomposition (p : Parameters) (r : RealHProgram.Regions) :
    iteration p r=iterationPrefix p r++[.ifThenElse (.constBool p.gateK)
      (gateKey p r++[RealHProgram.updateStmt p.toLayout p.swapped])
      (gateValue p r++[RealHProgram.updateStmt p.toLayout p.swapped])] := by
  by_cases h:p.swapped <;> reduce_iteration h

theorem iteration_prefix_ast (p : Parameters) (r : RealHProgram.Regions) :
    iterationPrefix p r=
      [.assign .blockPtr [p.BK,p.BT] "p_k" (keyPtr p r.key),
       .assign .blockPtr [p.BT,p.BV] "p_v" (valuePtr p r.value),
       .assign .blockPtr [p.BK,p.BV] "p_h" (historyPtr p r.history)]++
        (RealHProgram.iterationPrelude p.toLayout r).drop 3 := by
  by_cases h:p.swapped <;> reduce_iteration h

theorem gate_key_ast (p : Parameters) (r : RealHProgram.Regions) :
    gateKey p r=
      [.assign .blockPtr [p.BK,p.BT] "p_g" (keyPtr p r.gate),
       .assign .blockPtr [p.BK] "p_gn" (endpointPtr p r.gate "s_k_h" "s_k_t" "s_k_d" p.BK "i_k")]++
        (RealHProgram.gateKey p.toLayout r).drop 2 := by
  by_cases h:p.swapped <;> reduce_iteration h

theorem gate_value_ast (p : Parameters) (r : RealHProgram.Regions) :
    gateValue p r=
      [.assign .blockPtr [p.BT,p.BV] "p_g" (valuePtr p r.gate),
       .assign .blockPtr [p.BV] "p_gn" (endpointPtr p r.gate "s_v_h" "s_v_t" "s_v_d" p.BV "i_v")]++
        (RealHProgram.gateValue p.toLayout r).drop 2 := by
  by_cases h:p.swapped <;> reduce_iteration h

#axiomsClean iteration_decomposition
#axiomsClean iteration_prefix_ast
#axiomsClean gate_key_ast
#axiomsClean gate_value_ast
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
