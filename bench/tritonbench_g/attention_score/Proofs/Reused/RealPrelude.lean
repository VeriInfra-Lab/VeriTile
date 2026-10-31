import bench.tritonbench_g.attention_score.Proofs.Reused.RealPrepared
import bench.tritonbench_g.attention_score.Proofs.Reused.RealLoop

/-! Initialize the query-loop invariant from actual program coordinates and
initial input memory. Grouped-head addresses are mathematical Nat addresses;
legal launch divisors and physical allocation bounds belong to the IO contract. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealPrelude
open VeriTile.Triton RealFrame
set_option maxHeartbeats 1000000
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

/-- Initialize every loop-carried register from program coordinates and the
actual K load. No register contents or output equations are assumed. -/
theorem correct (s : BlockState) (Q K M : RegionName)
    (H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool) (keyRange : s.pids 0*B<2147483648) :
    ∃ t, stepStmts (RealProgram.prelude Q K M scale H HK N RN NK B D
      sqz sqh sqm sqk skz skh skn skk evenN) s = some t ∧
      RealLoop.Invariant (config s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk
        offset size scale window complement evenM evenN candidate) s 0 t ∧
      t.regs .nat [] "lo" = some (Tile.scalar 0) ∧
      t.regs .int [] "hi" = some (Tile.scalar (N:Int)) ∧
      t.regs .int [] "off_z" = some (Tile.scalar ((s.pids 1 / H:Nat):Int)) ∧
      t.regs .int [] "off_h" = some (Tile.scalar ((s.pids 1 % H:Nat):Int)) ∧
      t.regs .blockPtr [D,B] "K_block_ptr" = some (RealLoads.keyPointer K B D NK
        (s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh) skn skk (s.pids 0)) ∧
      Frame RealProgram.writes s t := by
  suffices he : ∃ t, stepStmts (RealProgram.prelude Q K M scale H HK N RN NK B D
      sqz sqh sqm sqk skz skh skn skk evenN) s = some t ∧
      RealLoop.Invariant (config s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk
        offset size scale window complement evenM evenN candidate) s 0 t ∧
      t.regs .nat [] "lo" = some (Tile.scalar 0) ∧
      t.regs .int [] "hi" = some (Tile.scalar (N:Int)) ∧
      t.regs .int [] "off_z" = some (Tile.scalar ((s.pids 1 / H:Nat):Int)) ∧
      t.regs .int [] "off_h" = some (Tile.scalar ((s.pids 1 % H:Nat):Int)) ∧
      t.regs .blockPtr [D,B] "K_block_ptr" = some (RealLoads.keyPointer K B D NK
        (s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh) skn skk (s.pids 0)) by
    obtain ⟨t,ht,hi,hlo,hhi,hz,hh,hk⟩ := he
    exact ⟨t,ht,hi,hlo,hhi,hz,hh,hk,(RealProgram.prelude_code Q K M scale H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk evenN).frame ht⟩
  let u := RealPrepared.prepared s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk
  have hp := RealPrepared.prepared_run s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk keyRange
  rw [RealPrefix.prelude_shape,ProgramRewrite.append,hp,Option.bind_some]
  change ∃ t,stepStmts (RealPrefix.afterKey B D N scale evenN) u=some t ∧ _
  unfold RealPrefix.afterKey
  rw [stepStmts.cons_some (RealIteration.load_key u Q K M B D NK
    ((s.pids 1 / H * sqz + s.pids 1 % H * sqh)) ((s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh))
    (s.pids 1 * RN) sqm sqk skn skk (s.pids 0) evenN (by simp [u,RealPrepared.prepared,RealLoads.keyPointer]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat 0 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.castNatToInt (.constNat N)) _ = some (Tile.scalar (N:Int)) from by simp only [evalOp.eq_def];rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_const scale _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul NumericDType.real Broadcast.nil (Op.ref .real [] "qk_scale")
        (Op.const 1.4426950408889634)) _
        = some (Tile.scalar (some (scale * 1.4426950408889634))) from by
      rw [evalOp_mul]
      simp only [evalOp_ref, evalOp_const, BlockState.setReg_same, Option.bind_eq_bind,
        Option.bind_some]
      rfl))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, by simp, by simp, by simp [u,RealPrepared.prepared],
    by simp [u,RealPrepared.prepared],by simp [u,RealPrepared.prepared,RealLoads.keyPointer]⟩
  constructor
  all_goals simp [config,RealLoop.Config.inputs,RealLoop.Config.column,RealFormula.column,
    RealLoads.inputs,RealLoads.queryPointer,RealLoads.normalizerPointers,RealTiles.key,
    u,RealPrepared.prepared]
  all_goals first | (funext r a; simp only [BlockState.setReg_mem]) | rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealPrelude
