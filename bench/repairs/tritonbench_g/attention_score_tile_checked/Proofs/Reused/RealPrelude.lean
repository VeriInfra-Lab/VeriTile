import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.IndexProjection
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealProgram
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealLoop

/-! Initialize the query-loop invariant from actual program coordinates and
initial input memory. Grouped-head addresses are mathematical Nat addresses;
legal launch divisors and physical allocation bounds belong to the IO contract. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealPrelude
open VeriTile.Triton RealFrame
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
noncomputable section

def config (s : BlockState) (Q K M : RegionName)
    (H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool) : RealLoop.Config where
  Q := Q
  K := K
  M := M
  B := B
  D := D
  N := N
  NK := NK
  baseQ := s.pids 1 / H * sqz + s.pids 1 % H * sqh
  baseK := s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh
  baseM := s.pids 1 * RN
  sqm := sqm
  sqk := sqk
  skn := skn
  skk := skk
  startN := s.pids 0
  offset := offset
  size := size
  scale := scale
  window := window
  complement := complement
  evenM := evenM
  evenN := evenN
  candidate := candidate

private theorem evalOp_constInt (n : Int) (s : BlockState) :
    evalOp (.constInt n) s = some (Tile.scalar n) := by rw [evalOp]

private theorem eval_floorDiv {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.bop h.floorDiv bc vx vy)) := by rw [evalOp]

private theorem eval_mod {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.mod h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.bop h.mod bc vx vy)) := by rw [evalOp]

private theorem eval_ptrAdd {a b shape} (bc : Broadcast a b shape)
    (x : Op .ptr a) (y : Op .nat b) (s : BlockState) :
    evalOp (.ptrAdd bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.ptrAdd bc vx vy)) := by rw [evalOp]

private theorem eval_ptrBase (r : RegionName) (s : BlockState) :
    evalOp (.ptrBase r) s = some (Tile.scalar (r.cast,0)) := by rw [evalOp]

private theorem scalarBop {dtype : TileDType}
    (f : TileCarrier dtype → TileCarrier dtype → TileCarrier dtype)
    (a b : TileCarrier dtype) : Tile.bop f Broadcast.nil (Tile.scalar a) (Tile.scalar b) =
      Tile.scalar (f a b) := rfl

private theorem makePointer (r : RegionName) (base : Op .nat [])
    (parent : List Nat) (block : TileShape) (strides : List Nat) (offsets : List Nat)
    (s : BlockState) (value : Nat) (he : evalOp base s = some (Tile.scalar value)) :
    evalOp (.makeBlockPtrDyn r base parent block strides offsets) s = some
      (⟨fun _ => { region := r, baseOffset := value, parentShape := parent, blockShape := block, strides := strides, offsets := offsets }⟩ : Tile .blockPtr block) := by
  rw [evalOp]
  simp only [he, bind, Option.bind]
  rfl

private theorem naturalScalar (value : Op .int []) (s : BlockState) (n : Nat)
    (he : evalOp value s = some (Tile.scalar (n : Int))) :
    evalOp (.castIntToNat value) s = some (Tile.scalar n) := by
  rw [evalOp, he]
  rfl

private theorem keyPointer (r : RegionName) (base off : Op .nat [])
    (D B NK skk skn : Nat) (s : BlockState) (b n : Nat)
    (hb : evalOp base s = some (Tile.scalar b))
    (hn : evalOp off s = some (Tile.scalar n)) :
    evalOp (.makeBlockPtrDynOffsets r base [D,NK] [D,B] [skk,skn]
      [.constNat 0,off]) s = some (⟨fun _ => {
        region := r
        baseOffset := b
        parentShape := [D,NK]
        blockShape := [D,B]
        strides := [skk,skn]
        offsets := [0,n] }⟩ : Tile .blockPtr [D,B]) := by
  rw [makeBlockPtr2_eval]
  simp only [hb, List.mapM_cons, List.mapM_nil, evalOp_constNat, hn,
    bind, Option.bind, pure, Tile.scalar_data]

/-- Initialize every loop-carried register from program coordinates and the
actual K load. No register contents or output equations are assumed. -/
theorem correct (s : BlockState) (Q K M : RegionName)
    (H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool)
    (keyOffsetBound : s.pids 0 * B < 2147483648) :
    ∃ t, stepStmts (RealProgram.prelude Q K M scale H HK N RN NK B D
      sqz sqh sqm sqk skz skh skn skk evenN) s = some t ∧
      RealLoop.Invariant (config s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk
        offset size scale window complement evenM evenN candidate) s 0 t ∧
      t.regs .int [] "lo" = some (Tile.scalar 0) ∧
      t.regs .int [] "hi" = some (Tile.scalar (N : Int)) ∧
      t.regs .int [] "off_z" = some (Tile.scalar ((s.pids 1 / H : Nat) : Int)) ∧
      t.regs .int [] "off_h" = some (Tile.scalar ((s.pids 1 % H : Nat) : Int)) ∧
      t.regs .blockPtr [D,B] "K_block_ptr" = some (RealLoads.keyPointer K B D NK
        (s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh) skn skk (s.pids 0)) ∧
      Frame RealProgram.writes s t := by
  suffices he : ∃ t, stepStmts (RealProgram.prelude Q K M scale H HK N RN NK B D
      sqz sqh sqm sqk skz skh skn skk evenN) s = some t ∧
      RealLoop.Invariant (config s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk
        offset size scale window complement evenM evenN candidate) s 0 t ∧
      t.regs .int [] "lo" = some (Tile.scalar 0) ∧
      t.regs .int [] "hi" = some (Tile.scalar (N : Int)) ∧
      t.regs .int [] "off_z" = some (Tile.scalar ((s.pids 1 / H : Nat) : Int)) ∧
      t.regs .int [] "off_h" = some (Tile.scalar ((s.pids 1 % H : Nat) : Int)) ∧
      t.regs .blockPtr [D,B] "K_block_ptr" = some (RealLoads.keyPointer K B D NK
        (s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh) skn skk (s.pids 0)) by
    obtain ⟨t,ht,hi,hlo,hhi,hz,hh,hk⟩ := he
    exact ⟨t,ht,hi,hlo,hhi,hz,hh,hk,(RealProgram.prelude_code Q K M scale H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk evenN).frame ht⟩
  unfold RealProgram.prelude
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.castNatToInt (.programId 0)) s = some (Tile.scalar ((s.pids 0 : Nat) : Int)) by simp [evalOp.eq_def, Tile.scalar, -Tile.scalar_eta]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.castNatToInt (.programId 1)) _ = some (Tile.scalar ((s.pids 1 : Nat) : Int)) by simp [evalOp.eq_def, Tile.scalar, -Tile.scalar_eta]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.truncDiv Broadcast.nil (Op.ref .int [] "off_hz") (Op.constInt H)) _
        = some (Tile.scalar ((s.pids 1 / H : Nat) : Int)) from by
      rw [eval_truncDiv]
      simp only [evalOp_ref, evalOp_constInt, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      simp only [scalarBop]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.truncMod Broadcast.nil (Op.ref .int [] "off_hz") (Op.constInt H)) _
        = some (Tile.scalar ((s.pids 1 % H : Nat) : Int)) from by
      rw [eval_truncMod]
      simp only [evalOp_ref, evalOp_constInt, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      simp only [scalarBop]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.truncDiv Broadcast.nil (Op.ref .int [] "off_h")
        (Op.truncDiv Broadcast.nil (Op.constInt H) (Op.constInt HK))) _
        = some (Tile.scalar (((s.pids 1 % H) / (H / HK) : Nat) : Int)) from by
      rw [eval_truncDiv, eval_truncDiv]
      simp only [evalOp_ref, evalOp_constInt, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      simp only [scalarBop]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.int Broadcast.nil
        (Op.mul NumericDType.int Broadcast.nil (Op.ref .int [] "off_z") (Op.constInt sqz))
        (Op.mul NumericDType.int Broadcast.nil (Op.ref .int [] "off_h") (Op.constInt sqh))) _
        = some (Tile.scalar ((s.pids 1 / H * sqz + s.pids 1 % H * sqh : Nat) : Int)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_constInt, evalOp_constNat, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      simp [scalarBop, NumericDType.add, NumericDType.mul, Int.ofNat_tdiv, Int.ofNat_tmod, Nat.cast_add, Nat.cast_mul]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.int Broadcast.nil
        (Op.mul NumericDType.int Broadcast.nil (Op.ref .int [] "off_z") (Op.constInt skz))
        (Op.mul NumericDType.int Broadcast.nil (Op.ref .int [] "off_hkv") (Op.constInt skh))) _
        = some (Tile.scalar ((s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh : Nat) : Int)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_constInt, evalOp_constNat, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      simp [scalarBop, NumericDType.add, NumericDType.mul, Int.ofNat_tdiv, Int.ofNat_tmod, Nat.cast_add, Nat.cast_mul]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M)
        (.castIntToNat (.add .int .scalarL
          (.mul .int .nil (.ref .int [] "off_hz") (.castNatToInt (.constNat RN)))
          (.castNatToInt (.arange B))))) _ =
        some (⟨fun idx : TileIndex [B] => (M.cast, s.pids 1 * RN + idx.1.val)⟩ : Tile .ptr [B]) from by
      simp only [evalOp.eq_def, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true, bind, Option.bind]
      congr 1
      apply Tile.ext
      intro idx
      simp [Tile.ptrAdd, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
        ← Int.natCast_add, ← Int.natCast_mul, -Nat.cast_add, -Nat.cast_mul, -Tile.scalar_eta]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [B] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [B] => some (0 : ℝ)⟩ : Tile .real [B]) from by
      simp [evalOp_full]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makePointer Q (.castIntToNat (Op.ref .int [] "q_offset")) [N, D] [B, D] [sqm, sqk] [0, 0] _
      ((s.pids 1 / H * sqz + s.pids 1 % H * sqh)) (naturalScalar _ _ _ (by simp))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.makeBlockPtrDynOffsets K (.castIntToNat (.ref .int [] "k_offset"))
        [D, NK] [D, B] [skk, skn]
        [.constNat 0, .castIntToNat (.intToSigned 32
          (.mul .int .nil (.ref .int [] "start_n") (.castNatToInt (.constNat B))))]) _ =
        some (⟨fun _ => {
            region := K
            baseOffset := s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh
            parentShape := [D, NK]
            blockShape := [D, B]
            strides := [skk, skn]
            offsets := [0, s.pids 0 * B] }⟩ : Tile .blockPtr [D, B]) from by
      apply keyPointer
      · exact naturalScalar _ _ _ (by simp)
      · apply IndexProjection.key_offset _ _ (s.pids 0 * B) _ keyOffsetBound
        simp [evalOp.eq_def, Tile.scalar, -Tile.scalar_eta, Tile.bop, NumericDType.mul,
          Broadcast.leftIndex, Broadcast.rightIndex]))]
  rw [stepStmts.cons_some (RealIteration.load_key _ Q K M B D NK
    ((s.pids 1 / H * sqz + s.pids 1 % H * sqh)) ((s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh))
    (s.pids 1 * RN) sqm sqk skn skk (s.pids 0) evenN (by simp [RealLoads.keyPointer]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.full [] (Op.constInt 0)) _ = some (Tile.scalar (0 : Int)) by simp [evalOp.eq_def, Tile.scalar, -Tile.scalar_eta]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.castNatToInt (Op.constNat N)) _ = some (Tile.scalar (N : Int)) by simp [evalOp.eq_def, Tile.scalar, -Tile.scalar_eta]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.const scale) _ = _ from evalOp_const scale _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul NumericDType.real Broadcast.nil (Op.ref .real [] "qk_scale")
        (Op.const 1.4426950408889634)) _
        = some (Tile.scalar (some (scale * 1.4426950408889634))) from by
      rw [evalOp_mul]
      simp only [evalOp_ref, evalOp_const, BlockState.setReg_same, Option.bind_eq_bind,
        Option.bind_some, scalarBop]
      refine congrArg some (congrArg Tile.scalar ?_)
      simp only [NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, by simp, by simp, by simp, by simp, by simp [RealLoads.keyPointer]⟩
  constructor
  all_goals simp [config, RealLoop.Config.inputs, RealLoop.Config.column, RealFormula.column,
    RealLoads.inputs, RealLoads.queryPointer, RealLoads.normalizerPointers, RealTiles.key]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealPrelude
