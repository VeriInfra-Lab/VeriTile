import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeRealProgram
import VeriTile.Triton.Float.IntegerErasure
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource.Real
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

local macro "reduce_erased_body" h:term : tactic => `(tactic|
  (dsimp only [algorithm,projection,Kernel.eraseDType]
   simp only [($h),Bool.false_eq_true,↓reduceDIte,↓reduceIte]
   simp only [ScalarArgument.integer,ScalarArgument.statement,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType_truncDiv,Op.eraseDType_truncMod,Op.eraseDType_castFloat,Op.eraseDType,
    MemAccess.eraseDType,MaskOpt.eraseDType,
    NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]))

def activePrelude (p : Parameters) (s o : RegionName) : List Stmt :=
  (activeBody p s o).take 8

def activePreludeExecution (p : Parameters) (s o : RegionName) (initial : BlockState) :
    {final : BlockState // stepStmts (activePrelude p s o)
      (preludeExecution p s o initial).val=some final} := by
  by_cases h:p.swapped
  all_goals
    refine ⟨?_,?_⟩
    swap
    unfold activePrelude activeBody guardStatement
    reduce_erased_body h
    simp only [List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,
      List.take_succ_cons,List.take_zero]
    simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
      VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]
    simp [-Tile.scalar_eta,preludeExecution,stepStmts,stepStmt,VeriTile.Triton.eval_truncDiv,
      VeriTile.Triton.eval_truncMod,VeriTile.Triton.evalOp,Tile.scalar,Tile.bop,
      Tile.expandDim,Option.bind_some,Option.bind_eq_bind,Option.map_some,
      Option.pure_def,BlockState.setReg_same,BlockState.setReg_ne_name]
    repeat' erw [evalOp_expandDim]
    simp [-Tile.scalar_eta,VeriTile.Triton.evalOp,Tile.scalar,Tile.bop,
      Tile.expandDim,Option.bind_some,Option.bind_eq_bind,Option.map_some,
      Option.pure_def,BlockState.setReg_same,BlockState.setReg_ne_name]
    rfl


theorem active_memory (p : Parameters) (s o : RegionName) (initial : BlockState) :
    (activePreludeExecution p s o initial).val.mem=initial.mem := by
  by_cases h:p.swapped <;> simp [activePreludeExecution,h,preludeExecution] <;> rfl

def suffix (p : Parameters) (s o : RegionName) : List Stmt :=
  (activeBody p s o).drop 8

theorem suffix_ast (p : Parameters) (s o : RegionName) :
    suffix p s o=
      [.assign .real [p.BT,p.BS] "b_s"
        (.load .real (.blockPtr (.ref .blockPtr [p.BT,p.BS] "p_s") [0,1]) .none .unspecified),
       .assign .real [p.BT,p.BS] "b_o"
        (.dot (batch:=[]) (.ref .real [p.BT,p.BT] "m_s") (.ref .real [p.BT,p.BS] "b_s")
          {allowTF32:=some Bool.false}),
       .store .real [p.BT,p.BS] (.blockPtr (.ref .blockPtr [p.BT,p.BS] "p_o") [0,1])
         (.ref .real [p.BT,p.BS] "b_o") .none] := by
  by_cases h:p.swapped
  all_goals
    unfold suffix activeBody guardStatement
    reduce_erased_body h
    simp only [List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.drop]
    simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
      VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]
    rfl

#axiomsClean activePreludeExecution
#axiomsClean active_memory
#axiomsClean suffix_ast
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource.Real
