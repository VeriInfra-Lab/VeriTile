import bench.tritonbench_g.attn_fwd_triton.Proofs.AttnFwdTriton_Verified_AccessSafety_AccessSafety
import bench.tritonbench_g.attn_fwd_triton.Proofs.Preparation
import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealSafetyParts

/-! Source-level memory safety of the changed helper fragments. Register-only
arithmetic is reused without conditions on its numerical values. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.SafetyParts
open VeriTile.Triton Memory Preparation AccessSafety
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealRegisterSafety
open VeriTile.Bench.TritonBenchG.AttentionScore
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def middle (BM BN BD : Nat) (second diagonal : Bool) : List Stmt :=
  [RealBody.score BM BN BD] ++ guard BM BN second (second || diagonal) ++
    RealBody.center BM BN ++ [RealBody.probability BM BN] ++ RealBody.rescale BM BN BD

def middleWrites : List RegName := ["qk","mask","mask_first","m_ij","p","l_ij","alpha","l_i","acc"]

theorem middle_frame (BM BN BD : Nat) (second diagonal : Bool) :
    RealFrame.Code middleWrites (middle BM BN BD second diagonal) := by
  cases second <;> cases diagonal
  all_goals
    dsimp [middle,guard,RealBody.score,RealBody.center,RealBody.probability,RealBody.rescale]
    repeat' first
      | apply RealFrame.Code.ifThenElse
      | apply RealFrame.Code.assign
      | apply RealFrame.Code.nil
      | simp [middleWrites,maskName,RealPreparation.maskName]

theorem middle_safe (BM BN BD : Nat) (second diagonal : Bool) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (middle BM BN BD second diagonal) s := by
  apply list_safe
  cases second <;> cases diagonal <;> simp [middle,guard,tail,RealBody.score,RealBody.scoreExpr,RealBody.center,RealBody.centerExpr,
    RealBody.probability,RealBody.rescale,maskExpr,Op.SafeAt.eq_def]

theorem tail_safe (BM BN BD : Nat) (second : Bool) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealBody.accumulate BM BN BD ++ advance BN BD second) s := by
  apply list_safe
  simp [RealBody.accumulate,advance,wide,Op.SafeAt.eq_def]

theorem source_split (BM BN BD D : Nat) (second diagonal : Bool) :
    sourceIteration BM BN BD D second diagonal =
      loadPrefix BD BN D second ++ middle BM BN BD second diagonal ++
      [valueLoad BN BD D second] ++ (RealBody.accumulate BM BN BD ++ advance BN BD second) := by
  cases second <;> cases diagonal <;> simp [sourceIteration,flatIteration,suffix,middle,List.append_assoc]

theorem loads_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BN BD start : Nat)
    (second : Bool) (hp : Pointers s c BN BD start second)
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hctx : s.regs .int [] "n_ctx" = some (Tile.scalar c.p.context))
    (hbK : KeyBound bounds c BN BD start) (hbS : ksbase c BN+start/BN < bounds c.p.KS) :
    Stmt.TraceSafeList bounds (loadPrefix BD BN c.p.headDim second) s := by
  let sn := s.setReg "start_n" .int [] (Tile.scalar start)
  let sm := sn.setReg "k_mask" .bool [BD,BN] (keyMaskTile BD BN c.p.context c.p.headDim start)
  have hm := eval_key_mask sn BD BN c.p.context c.p.headDim start (by simp [sn,hn]) (by simp [sn]) (by simp [sn,hctx])
  unfold loadPrefix
  apply cons_of_run (by simp [Stmt.TraceSafe,Op.SafeAt.eq_def])
    (stepStmt_assign_eq_some (by simpa only [evalOp_ref] using hs))
  apply cons_of_run (by simp [Stmt.TraceSafe,keyMask,tail,Op.SafeAt.eq_def]) (stepStmt_assign_eq_some hm)
  apply TraceProofs.assign_cons
    (key_safe bounds sm c BN BD start (keyName second)
      (by cases second <;> simpa [sm,sn,keyName,RealMemory.keyName] using hp.key) (by simp [sm]) hbK)
  intro tile
  apply Stmt.TraceSafeList.cons_intro
  · rw [Stmt.TraceSafe]
    exact scale_safe bounds (sm.setReg "k" .real [BD,BN] tile) c.p.KS (ksbase c BN+start/BN) (scaleName second)
      (by cases second <;> simpa [sm,sn,scaleName,RealMemory.scaleName,keyScalePointer] using hp.scale) hbS
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem iteration_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BM BN BD start : Nat)
    (second diagonal : Bool) (hp : Pointers s c BN BD start second)
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hctx : s.regs .int [] "n_ctx" = some (Tile.scalar c.p.context))
    (hbK : KeyBound bounds c BN BD start) (hbS : ksbase c BN+start/BN < bounds c.p.KS)
    (hbV : ValueBound bounds c BN BD start) :
    Stmt.TraceSafeList bounds (sourceIteration BM BN BD c.p.headDim second diagonal) s := by
  obtain ⟨sl,el,_,_,hstart,fl⟩ := loads_execute s c BN BD start second hp hn hs hctx
  rw [source_split,List.append_assoc,List.append_assoc]
  apply append_of_run (loads_safe bounds s c BN BD start second hp hn hs hctx hbK hbS) el
  apply append_safe _ sl (middle_safe BM BN BD second diagonal bounds sl)
  intro sm em
  have fm := (middle_frame BM BN BD second diagonal).frame em
  have hv : sm.regs .ptr [BN,BD] (valueName second) = some (valuePointer c BN BD start) :=
    (fm.regs _ _ _ (by cases second <;> simp [middleWrites,valueName,RealMemory.valueName])).trans
      ((fl.regs _ _ _ (by cases second <;> simp [valueName,RealMemory.valueName])).trans hp.value)
  have hkeys : sm.regs .int [BN] "offs_n" = some (Tile.vec (fun i => i.val)) :=
    (fm.regs _ _ _ (by simp [middleWrites])).trans ((fl.regs _ _ _ (by simp)).trans hn)
  have hcur : sm.regs .int [] "start_n" = some (Tile.scalar start) :=
    (fm.regs _ _ _ (by simp [middleWrites])).trans hstart
  have hcontext : sm.regs .int [] "n_ctx" = some (Tile.scalar c.p.context) :=
    (fm.regs _ _ _ (by simp [middleWrites])).trans ((fl.regs _ _ _ (by simp)).trans hctx)
  apply Stmt.TraceSafeList.cons_intro (value_safe bounds sm c BN BD start second hv hkeys hcur hcontext hbV)
  intro t _
  exact tail_safe BM BN BD second bounds t

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.SafetyParts
