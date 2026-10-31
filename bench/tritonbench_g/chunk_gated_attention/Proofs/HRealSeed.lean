import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealIndices
import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedInit
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles RealHLoads
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def statePtr (p : Parameters) (r : RegionName) : Op .blockPtr [p.BK,p.BV] :=
  .makeBlockPtrDyn r
    (.castIntToNat (.add .int .nil
      (.add .int .nil
        (.mul .int .nil (.mul .int .nil (.ref .int [] "i_bh") (.castNatToInt (.constNat p.K)))
          (.castNatToInt (.constNat p.V)))
        (.mul .int .nil (.mul .int .nil (.ref .int [] "i_k") (.castNatToInt (.constNat p.BK)))
          (.castNatToInt (.constNat p.V))))
      (.mul .int .nil (.ref .int [] "i_v") (.castNatToInt (.constNat p.BV)))))
    [p.BK,p.BV] [p.BK,p.BV] [p.V,1] [0,0]

theorem seed_ast (p : Parameters) (r : Regions) :
    seedStage p r=
      [.assign .real [p.BK,p.BV] "b_h" (.full [p.BK,p.BV] (.const 0)),
       .ifThen (.constBool p.hasSeed)
         [.assign .blockPtr [p.BK,p.BV] "p_h" (statePtr p r.seed),
          .assign .real [p.BK,p.BV] "b_h"
            (.add .real (.consSame (.consSame .nil)) (.ref .real [p.BK,p.BV] "b_h")
              (.load .real (.blockPtr (.ref .blockPtr [p.BK,p.BV] "p_h") [0,1]) .none .unspecified))]] := by
  unfold seedStage
  by_cases h:p.swapped
  all_goals
    dsimp only [algorithm,projection,Kernel.eraseDType]
    simp only [h,Bool.false_eq_true,↓reduceDIte,↓reduceIte]
    simp only [ScalarArgument.integer,ScalarArgument.statement,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType_truncDiv,Op.eraseDType_truncMod,Op.eraseDType_castFloat,Op.eraseDType,
      MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]
    simp only [List.drop,List.take_succ_cons,List.take_zero]
    simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
      VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]
    rfl

theorem state_pointer_eval (p : Parameters) (l : Launch) (r : RegionName) (u : BlockState)
    (hh : u.regs .int [] "i_bh"=some (Tile.scalar (l.head : Int)))
    (hk : u.regs .int [] "i_k"=some (Tile.scalar (l.key : Int)))
    (hv : u.regs .int [] "i_v"=some (Tile.scalar (l.value : Int))) :
    evalOp (statePtr p r) u=some (RebasedAccess.statePointer p.toLayout l r) := by
  simp only [statePtr,evalOp,hh,hk,hv,Tile.bop,Tile.scalar,NumericDType.add,NumericDType.mul,
    Option.bind_some,Option.bind_eq_bind,Option.pure_def]
  try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  rfl

def seedResult (p : Parameters) (r : Regions) (l : Launch) (a : Nat → ℝ) (u : BlockState) : BlockState :=
  let zero:=u.setReg "b_h" .real [p.BK,p.BV] (stateTile p.toLayout (fun _ _=>0))
  if p.hasSeed then
    (zero.setReg "p_h" .blockPtr [p.BK,p.BV] (RebasedAccess.statePointer p.toLayout l r.seed)).setReg
      "b_h" .real [p.BK,p.BV] (stateTile p.toLayout (seedData p.toLayout l a (fun _=>0)))
  else zero

theorem seed_computes (p : Parameters) (r : Regions) (l : Launch) (a : Nat → ℝ) (u : BlockState)
    (hK : l.key*p.BK+p.BK≤p.K) (hV : l.value*p.BV+p.BV≤p.V)
    (hh : u.regs .int [] "i_bh"=some (Tile.scalar (l.head : Int)))
    (hk : u.regs .int [] "i_k"=some (Tile.scalar (l.key : Int)))
    (hv : u.regs .int [] "i_v"=some (Tile.scalar (l.value : Int)))
    (hm : ∀ off,u.readMem r.seed off=a off) :
    stepStmts (seedStage p r) u=some (seedResult p r l a u) := by
  rw [seed_ast]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BK,p.BV] (.const 0)) u=
    some (stateTile p.toLayout (fun _ _=>0)) from by simp [evalOp_full,evalOp_const]; rfl))]
  by_cases hs:p.hasSeed
  · simp only [hs]
    rw [stepStmts.cons_some (st:=.ifThen (.constBool Bool.true) _) (by
      rw [RealHInit.if_true]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (state_pointer_eval p l r.seed _ (by simpa using hh) (by simpa using hk) (by simpa using hv)))]
      erw [stepStmts.cons_some (stepStmt_assign_eq_some (RebasedInit.add_seed p.toLayout l r.seed a _
        hK hV (by simpa using hm) (by simp) (by simp))),stepStmts.nil])]
    rw [stepStmts.nil]
    simp only [seedResult,hs,↓reduceIte]
  · simp only [hs]
    rw [stepStmts.cons_some (RealHInit.if_false _ _),stepStmts.nil]
    simp only [seedResult,hs,Bool.false_eq_true,↓reduceIte]


#axiomsClean seed_ast
#axiomsClean state_pointer_eval
#axiomsClean seed_computes
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
