import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeRealSuffix
import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeErasure
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource.Real
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def enabled (p : Parameters) (initial : BlockState) : Prop :=
  linearIndex p initial < p.heads*(p.T/p.BT)*(p.S/p.BS)
instance (p : Parameters) (initial : BlockState) : Decidable (enabled p initial) :=
  inferInstanceAs (Decidable (_ < _))

def guardCondition (p : Parameters) (s o : RegionName) : Op .bool [] :=
  match guardStatement p s o with
  | .ifThen condition _ => condition
  | _ => .constBool Bool.false

theorem guard_structure (p : Parameters) (s o : RegionName) :
    guardStatement p s o=.ifThen (guardCondition p s o) (activeBody p s o) := by
  by_cases h:p.swapped
  all_goals
    unfold guardCondition activeBody guardStatement
    dsimp only [algorithm,projection,Kernel.eraseDType]
    simp only [h,Bool.false_eq_true,↓reduceDIte,↓reduceIte]
    simp only [ScalarArgument.integer,ScalarArgument.statement,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType_truncDiv,Op.eraseDType_truncMod,
      Op.eraseDType_castFloat,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
      NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
      VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]
    try rfl

theorem guard_eval (p : Parameters) (s o : RegionName) (initial : BlockState) :
    evalOp (guardCondition p s o) (preludeExecution p s o initial).val=
      some (Tile.scalar (decide (enabled p initial))) := by
  by_cases h:p.swapped
  all_goals
    unfold guardCondition guardStatement
    dsimp only [algorithm,projection,Kernel.eraseDType]
    simp only [h,Bool.false_eq_true,↓reduceDIte,↓reduceIte]
    simp only [ScalarArgument.integer,ScalarArgument.statement,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType_truncDiv,Op.eraseDType_truncMod,
      Op.eraseDType_castFloat,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
      NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
      VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]
    simp only [List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
    simp [-Tile.scalar_eta,preludeExecution,evalOp,Tile.scalar,Tile.cop,Tile.bop,
      NumericDType.mul,NumericDType.add,ComparableDType.lt,enabled,linearIndex,
      Int.natCast_add,Int.natCast_mul]
    simp only [←Int.natCast_ediv,←Int.natCast_mul,←Int.natCast_add]
    simp only [Int.ofNat_lt]
    try rfl

theorem active_computes (p : Parameters) (s o : RegionName) (initial : BlockState) :
    stepStmts (activeBody p s o) (preludeExecution p s o initial).val=
      some (activeResult p s o initial) := by
  rw [←List.take_append_drop 8 (activeBody p s o)]
  exact stepStmts.append_some (activePreludeExecution p s o initial).property
    |>.trans (suffix_computes p s o initial)

def footprint (p : Parameters) (o : RegionName) (initial : BlockState) : WriteFootprint :=
  fun cell=>enabled p initial ∧
    RealCumExecution.footprint (tileLayout p initial) localLaunch 0 o cell

theorem executes (p : Parameters) (dtype out_dtype : FloatDType) (s o : RegionName) (initial : BlockState)
    (hinj : enabled p initial →
      RealCumExecution.NoCollision (tileLayout p initial) localLaunch 0) :
    ∃ final, exec (kernel p dtype out_dtype s o).eraseDType.toAlgKernel initial=some final ∧
      (enabled p initial → ∀ i : TileIndex [p.BT,p.BS],
        final.readMem o (RealCumExecution.offset (tileLayout p initial) localLaunch 0 i)=
          RealCumExecution.expected (tileLayout p initial) localLaunch 0 (initial.readMem s) i) ∧
      BlockState.WriteWithin (footprint p o initial) initial final := by
  rw [erase_connection,erased_algorithm,exec,decomposition,
    stepStmts.append_some (preludeExecution p s o initial).property,guard_structure]
  by_cases ha : enabled p initial
  · refine ⟨activeResult p s o initial,?_,?_,?_⟩
    · simp only [stepStmts,stepStmt,guard_eval,Tile.scalar,Option.bind_some,Option.bind_eq_bind,
        decide_eq_true ha,↓reduceIte,active_computes]
    · intro _ i
      exact active_readback p s o initial (hinj ha) i
    · intro region off hn
      exact active_frame p s o initial region off (fun h=>hn ⟨ha,h⟩)
  · refine ⟨(preludeExecution p s o initial).val,?_,fun h=>False.elim (ha h),?_⟩
    · simp only [stepStmts,stepStmt,guard_eval,Tile.scalar,Option.bind_some,Option.bind_eq_bind,
        decide_eq_false ha,Bool.false_eq_true,↓reduceIte]
    · intro region off _
      rw [prelude_memory]

#axiomsClean guard_eval
#axiomsClean active_computes
#axiomsClean executes
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource.Real
