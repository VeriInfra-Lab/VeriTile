import bench.tritonbench_g.attention_score.Proofs.Reused.RealPrefix
import VeriTile.Triton.Semantics.UncheckedBlockLoad
import VeriTile.Triton.Float.SignedCast
import VeriTile.Triton.Math.IntegerIndex
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealPrepared
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 400000
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def prepared (s : BlockState) (Q K M : RegionName)
    (H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk : Nat) : BlockState :=
  let pm := s.pids 0
  let ph := s.pids 1
  let z := ph/H
  let h := ph%H
  let kh := h/(H/HK)
  let qb := z*sqz+h*sqh
  let kb := z*skz+kh*skh
  let s0 := s.setReg "start_n" .int [] (Tile.scalar (pm:Int))
  let s1 := s0.setReg "off_hz" .int [] (Tile.scalar (ph:Int))
  let s2 := s1.setReg "off_z" .int [] (Tile.scalar (z:Int))
  let s3 := s2.setReg "off_h" .int [] (Tile.scalar (h:Int))
  let s4 := s3.setReg "off_hkv" .int [] (Tile.scalar (kh:Int))
  let s5 := s4.setReg "q_offset" .int [] (Tile.scalar (qb:Int))
  let s6 := s5.setReg "k_offset" .int [] (Tile.scalar (kb:Int))
  let s7 := s6.setReg "m_ptrs" .ptr [B]
    (⟨fun i => (M,ph*RN+i.1.val)⟩ : Tile .ptr [B])
  let s8 := s7.setReg "o" .real [B] (⟨fun _ => some 0⟩ : Tile .real [B])
  let s9 := s8.setReg "Q_block_ptr" .blockPtr [B,D]
    (⟨fun _ => BlockPtr.mk Q qb [N,D] [B,D] [sqm,sqk] [0,0]⟩ : Tile .blockPtr [B,D])
  let s10 := s9.setReg "K_block_ptr" .blockPtr [D,B]
    (⟨fun _ => BlockPtr.mk K kb [D,NK] [D,B] [skk,skn] [0,pm*B]⟩ : Tile .blockPtr [D,B])
  s10

theorem prepared_run (s : BlockState) (Q K M : RegionName)
    (H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk : Nat)
    (keyRange : s.pids 0*B<2147483648) :
    stepStmts (RealPrefix.beforeKey Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk) s =
      some (prepared s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk) := by
  have nonneg : 0≤(s.pids 0:Int)*(B:Int) := Int.mul_nonneg (Int.natCast_nonneg _) (Int.natCast_nonneg _)
  have upper : (s.pids 0:Int)*(B:Int)<2147483648 := by exact_mod_cast keyRange
  have narrow : (BitVec.ofInt 32 ((s.pids 0:Int)*(B:Int))).toInt=(s.pids 0:Int)*(B:Int) := by
    apply BitVec.toInt_ofInt_eq_self (by decide)
    · norm_num; omega
    · norm_num; exact upper
  let pm := s.pids 0
  let ph := s.pids 1
  let z := ph/H
  let h := ph%H
  let kh := h/(H/HK)
  let qb := z*sqz+h*sqh
  let kb := z*skz+kh*skh
  let s0 := s.setReg "start_n" .int [] (Tile.scalar (pm:Int))
  let s1 := s0.setReg "off_hz" .int [] (Tile.scalar (ph:Int))
  let s2 := s1.setReg "off_z" .int [] (Tile.scalar (z:Int))
  let s3 := s2.setReg "off_h" .int [] (Tile.scalar (h:Int))
  let s4 := s3.setReg "off_hkv" .int [] (Tile.scalar (kh:Int))
  let s5 := s4.setReg "q_offset" .int [] (Tile.scalar (qb:Int))
  let s6 := s5.setReg "k_offset" .int [] (Tile.scalar (kb:Int))
  let s7 := s6.setReg "m_ptrs" .ptr [B]
    (⟨fun i => (M,ph*RN+i.1.val)⟩ : Tile .ptr [B])
  let s8 := s7.setReg "o" .real [B] (⟨fun _ => some 0⟩ : Tile .real [B])
  let s9 := s8.setReg "Q_block_ptr" .blockPtr [B,D]
    (⟨fun _ => BlockPtr.mk Q qb [N,D] [B,D] [sqm,sqk] [0,0]⟩ : Tile .blockPtr [B,D])
  let s10 := s9.setReg "K_block_ptr" .blockPtr [D,B]
    (⟨fun _ => BlockPtr.mk K kb [D,NK] [D,B] [skk,skn] [0,pm*B]⟩ : Tile .blockPtr [D,B])
  let keyIndex : Op .nat [] := .castIntToNat (Op.intToSigned 32 (.mul .int .nil (.ref .int [] "start_n") (.castNatToInt (.constNat B))))
  have he0 : stepStmt (.assign .int [] "start_n" (.castNatToInt (.programId 0))) s = some s0 := by
    apply stepStmt_assign_eq_some
    simp [evalOp.eq_def,Option.bind,
      s0,pm,ph,z,h,kh,qb,kb,
      Tile.bop,Tile.cop,Tile.scalar,Tile.ptrAdd,Tile.select,
      Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,
      IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,Op.tritonDiv,Op.truncDiv,Op.tritonMod,Op.truncMod,
      Region.cast,IntegerExpr.sameShape,narrow,-Tile.scalar_eta]
    all_goals try rfl

  have he1 : stepStmt (.assign .int [] "off_hz" (.castNatToInt (.programId 1))) s0 = some s1 := by
    apply stepStmt_assign_eq_some
    simp [evalOp.eq_def,Option.bind,
      s1,s0,pm,ph,z,h,kh,qb,kb,
      Tile.bop,Tile.cop,Tile.scalar,Tile.ptrAdd,Tile.select,
      Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,
      IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,Op.tritonDiv,Op.truncDiv,Op.tritonMod,Op.truncMod,
      Region.cast,IntegerExpr.sameShape,narrow,-Tile.scalar_eta]
    all_goals try rfl

  have he2 : stepStmt (.assign .int [] "off_z" (.tritonDiv .int .nil (.ref .int [] "off_hz") (.castNatToInt (.constNat H)))) s1 = some s2 := by
    apply stepStmt_assign_eq_some
    simp [evalOp.eq_def,Option.bind,
      s2,s1,s0,pm,ph,z,h,kh,qb,kb,
      Tile.bop,Tile.cop,Tile.scalar,Tile.ptrAdd,Tile.select,
      Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,
      IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,Op.tritonDiv,Op.truncDiv,Op.tritonMod,Op.truncMod,
      Region.cast,IntegerExpr.sameShape,narrow,-Tile.scalar_eta]
    all_goals try rfl

  have he3 : stepStmt (.assign .int [] "off_h" (.tritonMod .int .nil (.ref .int [] "off_hz") (.castNatToInt (.constNat H)))) s2 = some s3 := by
    apply stepStmt_assign_eq_some
    simp [evalOp.eq_def,Option.bind,
      s3,s2,s1,s0,pm,ph,z,h,kh,qb,kb,
      Tile.bop,Tile.cop,Tile.scalar,Tile.ptrAdd,Tile.select,
      Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,
      IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,Op.tritonDiv,Op.truncDiv,Op.tritonMod,Op.truncMod,
      Region.cast,IntegerExpr.sameShape,narrow,-Tile.scalar_eta]
    all_goals try rfl

    funext i
    simp only [not_lt.mpr (Int.natCast_nonneg _),↓reduceIte,Int.emod_def]
  have he4 : stepStmt (.assign .int [] "off_hkv" (.tritonDiv .int .nil (.ref .int [] "off_h")
     (.castNatToInt (.tritonDiv .nat .nil (.constNat H) (.constNat HK))))) s3 = some s4 := by
    apply stepStmt_assign_eq_some
    simp [evalOp.eq_def,Option.bind,
      s4,s3,s2,s1,s0,pm,ph,z,h,kh,qb,kb,
      Tile.bop,Tile.cop,Tile.scalar,Tile.ptrAdd,Tile.select,
      Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,
      IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,Op.tritonDiv,Op.truncDiv,Op.tritonMod,Op.truncMod,
      Region.cast,IntegerExpr.sameShape,narrow,-Tile.scalar_eta]
    all_goals try rfl

  have he5 : stepStmt (.assign .int [] "q_offset" (.add .int .nil
     (.mul .int .nil (.ref .int [] "off_z") (.castNatToInt (.constNat sqz)))
     (.mul .int .nil (.ref .int [] "off_h") (.castNatToInt (.constNat sqh))))) s4 = some s5 := by
    apply stepStmt_assign_eq_some
    simp [evalOp.eq_def,Option.bind,
      s5,s4,s3,s2,s1,s0,pm,ph,z,h,kh,qb,kb,
      Tile.bop,Tile.cop,Tile.scalar,Tile.ptrAdd,Tile.select,
      Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,
      IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,Op.tritonDiv,Op.truncDiv,Op.tritonMod,Op.truncMod,
      Region.cast,IntegerExpr.sameShape,narrow,-Tile.scalar_eta]
    all_goals try rfl

  have he6 : stepStmt (.assign .int [] "k_offset" (.add .int .nil
     (.mul .int .nil (.ref .int [] "off_z") (.castNatToInt (.constNat skz)))
     (.mul .int .nil (.ref .int [] "off_hkv") (.castNatToInt (.constNat skh))))) s5 = some s6 := by
    apply stepStmt_assign_eq_some
    simp [evalOp.eq_def,Option.bind,
      s6,s5,s4,s3,s2,s1,s0,pm,ph,z,h,kh,qb,kb,
      Tile.bop,Tile.cop,Tile.scalar,Tile.ptrAdd,Tile.select,
      Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,
      IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,Op.tritonDiv,Op.truncDiv,Op.tritonMod,Op.truncMod,
      Region.cast,IntegerExpr.sameShape,narrow,-Tile.scalar_eta]
    all_goals try rfl

  have he7 : stepStmt (.assign .ptr [B] "m_ptrs" (.ptrAdd .scalarL (.ptrBase M)
     (.castIntToNat (.add .int .scalarL
       (.mul .int .nil (.ref .int [] "off_hz") (.castNatToInt (.constNat RN)))
       (.castNatToInt (.arange B)))))) s6 = some s7 := by
    apply stepStmt_assign_eq_some
    simp [evalOp.eq_def,Option.bind,
      s7,s6,s5,s4,s3,s2,s1,s0,pm,ph,z,h,kh,qb,kb,
      Tile.bop,Tile.cop,Tile.scalar,Tile.ptrAdd,Tile.select,
      Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,
      IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,Op.tritonDiv,Op.truncDiv,Op.tritonMod,Op.truncMod,
      Region.cast,IntegerExpr.sameShape,narrow,-Tile.scalar_eta]
    all_goals try rfl

  have he8 : stepStmt (.assign .real [B] "o" (.full [B] (.const 0))) s7 = some s8 := by
    apply stepStmt_assign_eq_some
    simp [evalOp.eq_def,Option.bind,
      s8,s7,s6,s5,s4,s3,s2,s1,s0,pm,ph,z,h,kh,qb,kb,
      Tile.bop,Tile.cop,Tile.scalar,Tile.ptrAdd,Tile.select,
      Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,
      IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,Op.tritonDiv,Op.truncDiv,Op.tritonMod,Op.truncMod,
      Region.cast,IntegerExpr.sameShape,narrow,-Tile.scalar_eta]
    all_goals try rfl

  have he9 : stepStmt (.assign .blockPtr [B,D] "Q_block_ptr"
     (.makeBlockPtrDyn Q (.castIntToNat (.ref .int [] "q_offset")) [N,D] [B,D] [sqm,sqk] [0,0])) s8 = some s9 := by
    apply stepStmt_assign_eq_some
    simp [evalOp.eq_def,Option.bind,
      s9,s8,s7,s6,s5,s4,s3,s2,s1,s0,pm,ph,z,h,kh,qb,kb,
      Tile.bop,Tile.cop,Tile.scalar,Tile.ptrAdd,Tile.select,
      Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,
      IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,Op.tritonDiv,Op.truncDiv,Op.tritonMod,Op.truncMod,
      Region.cast,IntegerExpr.sameShape,narrow,-Tile.scalar_eta]
    all_goals try rfl

  have he10 : stepStmt (.assign .blockPtr [D,B] "K_block_ptr"
     (.makeBlockPtrDynOffsets K (.castIntToNat (.ref .int [] "k_offset")) [D,NK] [D,B] [skk,skn]
       [.constNat 0,keyIndex])) s9 = some s10 := by
    apply stepStmt_assign_eq_some
    have ek : evalOp keyIndex s9=some (Tile.scalar (pm*B)) :=
      RealPrefix.key_index_eval s9 pm B (by simp [s9,s8,s7,s6,s5,s4,s3,s2,s1,s0]) keyRange
    change evalOp (.makeBlockPtrDynOffsets K (.castIntToNat (.ref .int [] "k_offset")) [D,NK] [D,B] [skk,skn] [.constNat 0,keyIndex]) s9 = _
    rw [makeBlockPtr2_eval]
    simp only [List.mapM_cons,List.mapM_nil,evalOp_constNat,ek]
    simp [evalOp.eq_def,Option.bind,
      s10,s9,s8,s7,s6,s5,s4,s3,s2,s1,s0,pm,ph,z,h,kh,qb,kb,
      Tile.bop,Tile.cop,Tile.scalar,Tile.ptrAdd,Tile.select,
      Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,
      IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,Op.tritonDiv,Op.truncDiv,Op.tritonMod,Op.truncMod,
      Region.cast,IntegerExpr.sameShape,narrow,-Tile.scalar_eta]
    all_goals try rfl

  change stepStmts _ s = some s10
  simp only [RealPrefix.beforeKey]
  rw [stepStmts.cons_some he0,
      stepStmts.cons_some he1,
      stepStmts.cons_some he2,
      stepStmts.cons_some he3,
      stepStmts.cons_some he4,
      stepStmts.cons_some he5,
      stepStmts.cons_some he6,
      stepStmts.cons_some he7,
      stepStmts.cons_some he8,
      stepStmts.cons_some he9,
      stepStmts.cons_some he10,stepStmts.nil]
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealPrepared
