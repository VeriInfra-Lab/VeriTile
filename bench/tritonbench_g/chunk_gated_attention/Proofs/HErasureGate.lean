import bench.tritonbench_g.chunk_gated_attention.Proofs.HProjection
import VeriTile.Triton.Float.IntegerErasure
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 600000
set_option linter.unusedSimpArgs false
theorem erase_gate (p : Parameters) (r : RealHProgram.Regions)
    (dot gate history seed final : FloatDType) :
    (projection p r dot gate history seed final).val.eraseDType=
      (projection p r dot .real history seed final).val.eraseDType := by
  by_cases h:p.swapped
  all_goals
    dsimp only [projection,Kernel.eraseDType]
    simp only [h,Bool.false_eq_true,↓reduceDIte,↓reduceIte]
    cases gate <;>
      simp only [ScalarArgument.integer,ScalarArgument.statement,Stmt.eraseDTypeList,Stmt.eraseDType,
        Op.eraseDType_truncDiv,Op.eraseDType_truncMod,Op.eraseDType_castFloat,Op.eraseDType,
        MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
        IntegralDType.eraseDType,VeriTile.Triton.eraseDType,FloatDType.toTileDType,
        List.map_cons,List.map_nil] <;>
      simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
        NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
        VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil] <;> rfl

#axiomsClean erase_gate
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource
