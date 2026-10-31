import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealPointers
import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedStore
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHFinal
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

theorem final_ast (p : Parameters) (r : RealHProgram.Regions) :
    finalStage p r=[.ifThen (.constBool p.storeFinal)
      [.assign .blockPtr [p.BK,p.BV] "p_h" (statePtr p r.final),RealHProgram.storeCarry p.toLayout]] := by
  unfold finalStage
  by_cases h:p.swapped
  all_goals
    dsimp only [algorithm,projection,Kernel.eraseDType]
    simp only [h,Bool.false_eq_true,↓reduceDIte,↓reduceIte]
    simp only [ScalarArgument.integer,ScalarArgument.statement,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType_truncDiv,Op.eraseDType_truncMod,Op.eraseDType_castFloat,Op.eraseDType,
      MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]
    simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
      VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]
    rfl

namespace Final
open RealHMath RealHProgram RealHTiles
abbrev base := RealHFinal.base
abbrev footprint := RealHFinal.footprint

def output (p : Parameters) (l : Launch) (r : Regions) (storeFinal : Bool)
    (f : StateValue p.toLayout) (u : BlockState) : BlockState :=
  if storeFinal then RealHStore.scatter p.toLayout l r.final (base p.toLayout l) p.V 1 f
    (u.setReg "p_h" .blockPtr [p.BK,p.BV] (RebasedAccess.statePointer p.toLayout l r.final))
  else u

theorem executes (p : Parameters) (l : Launch) (r : Regions)
    (f : StateValue p.toLayout) (u : BlockState)
    (hk : l.key*p.BK+p.BK≤p.K) (hv : l.value*p.BV+p.BV≤p.V)
    (hik : u.regs .int [] "i_k"=some (Tile.scalar (l.key : Int)))
    (hiv : u.regs .int [] "i_v"=some (Tile.scalar (l.value : Int)))
    (hih : u.regs .int [] "i_bh"=some (Tile.scalar (l.head : Int)))
    (hh : u.regs .real [p.BK,p.BV] "b_h"=some (stateTile p.toLayout f)) :
    stepStmts (finalStage p r) u=some (output p l r p.storeFinal f u) := by
  rw [final_ast]
  by_cases h:p.storeFinal
  · simp only [h]
    rw [stepStmts.cons_some (st:=.ifThen (.constBool Bool.true) _) (by
      rw [RealHInit.if_true]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some (state_pointer_eval p l r.final u hih hik hiv))]
      erw [stepStmts.cons_some (RebasedStore.store_eval p.toLayout l r.final (base p.toLayout l) p.V 1 f "p_h" _ hk hv
        (by simp [hh]) (by simp [RebasedAccess.statePointer,RebasedStore.shifted,base,RealHFinal.base])),stepStmts.nil])]
    rw [stepStmts.nil]
    rfl
  · simp only [h]
    rw [stepStmts.cons_some (RealHInit.if_false _ _),stepStmts.nil]
    rfl

theorem readback (p : Parameters) (l : Launch) (r : Regions) (f : StateValue p.toLayout) (u : BlockState)
    (hinj : RealHStore.NoCollision p.toLayout l (base p.toLayout l) p.V 1)
    (e : Fin p.BK) (j : Fin p.BV) (ha : RealHStore.active p.toLayout l e.val j.val) :
    (output p l r Bool.true f u).readMem r.final
      (stateAddress p.toLayout l (RealHMath.keyIndex p.toLayout l e.val) (RealHMath.valueIndex p.toLayout l j.val))=f e j := by
  simpa [output,RealHStore.address,base,stateAddress] using
    RealHStore.readback p.toLayout l r.final (base p.toLayout l) p.V 1 f _ hinj e j ha

theorem frame (p : Parameters) (l : Launch) (r : Regions) (storeFinal : Bool)
    (f : StateValue p.toLayout) (u : BlockState) :
    BlockState.WriteWithin (footprint p.toLayout l r storeFinal) u (output p l r storeFinal f u) := by
  cases storeFinal
  · intro _ _ _;rfl
  · intro region off hn
    apply RealHStore.frame p.toLayout l r.final (base p.toLayout l) p.V 1 f
      (u.setReg "p_h" .blockPtr [p.BK,p.BV] (RebasedAccess.statePointer p.toLayout l r.final)) region off
    intro h
    exact hn ⟨rfl,h⟩

#axiomsClean executes
#axiomsClean readback
#axiomsClean frame
end Final
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
