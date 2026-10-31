import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeRealProgram
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem erased_algorithm (p : Parameters) (dtype out_dtype : FloatDType) (s o : RegionName) :
    (projection p dtype out_dtype s o).val.eraseDType=Real.algorithm p s o := by
  by_cases h:p.swapped
  all_goals
    dsimp only [Real.algorithm,projection,Kernel.eraseDType]
    simp only [h,Bool.false_eq_true,↓reduceDIte,↓reduceIte]
    cases dtype <;> cases out_dtype <;>
      simp only [ScalarArgument.integer,ScalarArgument.statement,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType_truncDiv,Op.eraseDType_truncMod,
        Op.eraseDType_castFloat,Op.eraseDType,
        MemAccess.eraseDType,MaskOpt.eraseDType,
        NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
        VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil] <;>
      simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
        NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
        VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil] <;> rfl

#axiomsClean erased_algorithm
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource
