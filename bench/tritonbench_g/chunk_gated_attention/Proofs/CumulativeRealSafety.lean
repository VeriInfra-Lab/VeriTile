import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeRealExecution
import VeriTile.Triton.Memory.Integer
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHAccessSafety
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource.Real
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

structure Bounds (p : Parameters) (s o : RegionName) (initial : BlockState)
    (bounds : RegionBounds) : Prop where
  input : RealHAccessSafety.MatrixBounds s (tileBase p initial) p.BT p.BS p.BT p.BS
    p.timeStride p.featureStride 0 0 bounds
  output : RealHAccessSafety.MatrixBounds o (tileBase p initial) p.BT p.BS p.BT p.BS
    p.timeStride p.featureStride 0 0 bounds

local macro "reduce_body_for_safety" h:term : tactic => `(tactic|
  (dsimp only [algorithm,projection,Kernel.eraseDType]
   simp only [($h),Bool.false_eq_true,↓reduceDIte,↓reduceIte]
   simp only [ScalarArgument.integer,ScalarArgument.statement,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType_truncDiv,Op.eraseDType_truncMod,
    Op.eraseDType_castFloat,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]))

theorem prelude_safe (p : Parameters) (s o : RegionName) (bounds : RegionBounds) (u : BlockState) :
    Stmt.TraceSafeList bounds (prelude p s o) u := by
  rw [prelude_ast]
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

theorem activePrelude_safe (p : Parameters) (s o : RegionName) (bounds : RegionBounds) (u : BlockState) :
    Stmt.TraceSafeList bounds (activePrelude p s o) u := by
  unfold activePrelude activeBody guardStatement
  by_cases h:p.swapped
  all_goals
    reduce_body_for_safety h
    simp only [h,Bool.false_eq_true,↓reduceIte,List.getElem?_cons_succ,List.getElem?_cons_zero,
      Option.getD_some,List.take_succ_cons,List.take_zero]
    simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
      VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]
    apply TraceProofs.all_safe
    simp [Stmt.TraceSafe,Op.safeAt_truncDiv,Op.safeAt_truncMod,Op.SafeAt]
    simp [Op.SafeAt.eq_def]

theorem suffix_safe (p : Parameters) (s o : RegionName) (bounds : RegionBounds) (initial : BlockState)
    (hb : Bounds p s o initial bounds) :
    Stmt.TraceSafeList bounds (suffix p s o) (activePreludeExecution p s o initial).val := by
  rw [suffix_ast]
  apply TraceProofs.assign_cons
  · apply RealHAccessSafety.load2_safe s (tileBase p initial) p.BT p.BS p.BT p.BS
      p.timeStride p.featureStride 0 0 "p_s" bounds _ hb.input
    simpa [tileLayout,localLaunch,
      VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealStep.pointer,
      RealHPointers.matrixPointer] using active_input_pointer p s o initial
  intro bs
  apply TraceProofs.assign_cons
  · simp [Op.SafeAt,Op.SafeAt.eq_def]
  intro bo
  apply Stmt.TraceSafeList.cons_intro
    (RealHAccessSafety.store2_safe o (tileBase p initial) p.BT p.BS p.BT p.BS
      p.timeStride p.featureStride 0 0 "p_o" "b_o" bounds _ hb.output
      (by simpa [RealHPointers.matrixPointer] using active_output_pointer p s o initial))
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

theorem active_safe (p : Parameters) (s o : RegionName) (bounds : RegionBounds) (initial : BlockState)
    (hb : Bounds p s o initial bounds) :
    Stmt.TraceSafeList bounds (activeBody p s o) (preludeExecution p s o initial).val := by
  rw [←List.take_append_drop 8 (activeBody p s o)]
  apply TraceProofs.append_safe
  · exact activePrelude_safe p s o bounds _
  · intro final he
    change stepStmts (activePrelude p s o) (preludeExecution p s o initial).val=some final at he
    rw [(activePreludeExecution p s o initial).property] at he
    cases he
    exact suffix_safe p s o bounds initial hb

theorem guardCondition_safe (p : Parameters) (s o : RegionName) (bounds : RegionBounds) (u : BlockState) :
    (guardCondition p s o).SafeAt bounds u := by
  unfold guardCondition guardStatement
  by_cases h:p.swapped
  all_goals
    reduce_body_for_safety h
    simp only [h,Bool.false_eq_true,↓reduceIte,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
    simp [Op.SafeAt,Op.SafeAt.eq_def]

theorem source_safe (p : Parameters) (dtype out_dtype : FloatDType) (s o : RegionName)
    (bounds : RegionBounds) (initial : BlockState)
    (hb : enabled p initial → Bounds p s o initial bounds) :
    (kernel p dtype out_dtype s o).eraseDType.toAlgKernel.TraceSafe bounds initial := by
  rw [erase_connection,erased_algorithm,Kernel.TraceSafe,decomposition]
  apply TraceProofs.append_safe
  · exact prelude_safe p s o bounds initial
  · intro final he
    rw [(preludeExecution p s o initial).property] at he
    cases he
    rw [guard_structure]
    apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
    rw [Stmt.TraceSafe]
    refine ⟨guardCondition_safe p s o bounds _,?_⟩
    rw [guard_eval]
    by_cases ha:enabled p initial
    · simp only [Tile.scalar,decide_eq_true ha,↓reduceIte]
      exact active_safe p s o bounds initial (hb ha)
    · simp only [Tile.scalar,decide_eq_false ha,Bool.false_eq_true,↓reduceIte]

theorem source_flatten_ok (p : Parameters) (dtype out_dtype : FloatDType) (s o : RegionName) :
    (kernel p dtype out_dtype s o).eraseDType.toAlgKernel.FlattenOk := by
  rw [erase_connection,erased_algorithm,Kernel.FlattenOk]
  by_cases h:p.swapped
  all_goals
    reduce_body_for_safety h
    simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
      VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]
    simp [StmtList.FlattenOk,Stmt.FlattenOk,Op.flattenOk_truncDiv,Op.flattenOk_truncMod,Op.FlattenOk]
    simp [Op.FlattenOk.eq_def]

#axiomsClean source_safe
#axiomsClean source_flatten_ok
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource.Real
