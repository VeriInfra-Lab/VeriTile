import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.RebasedCumAccess

/-! Complete repaired cumulative execution, including flattened launch
initialization and exhausted autotune batches. The actual loaded/stored prefix
and tagged-cell frame use the existing local triangular-dot proof. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumExecution
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG
open ChunkGatedAttention
open RealCumProgram
open ChunkCumsumVector (lowerTriTile)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def linear (base : Nat) (u : BlockState) := u.pids 0+base

def decoded (p : Layout) (base : Nat) (u : BlockState) : Launch :=
  ⟨linear base u % (p.S/p.BS), linear base u / ((p.S/p.BS)*(p.T/p.BT))⟩

def chunk (p : Layout) (base : Nat) (u : BlockState) :=
  (linear base u / (p.S/p.BS)) % (p.T/p.BT)

def linearState (base : Nat) (u : BlockState) := u.setReg "linear" .nat [] (Tile.scalar (linear base u))

def initial (p : Layout) (base : Nat) (u : BlockState) : BlockState :=
  let l := decoded p base u
  let s1 := (linearState base u).setReg "i_s" .nat [] (Tile.scalar l.feature)
  let s2 := s1.setReg "i_t" .nat [] (Tile.scalar (chunk p base u))
  let s3 := s2.setReg "i_bh" .nat [] (Tile.scalar l.head)
  let s4 := s3.setReg "o_i" .nat [p.BT] ⟨fun i => i.1.val⟩
  s4.setReg "m_s" .real [p.BT,p.BT] (lowerTriTile p.BT)

theorem initializes (p : Layout) (base : Nat) (u : BlockState) :
    stepStmts (RebasedCumProgram.decodeCode p ++ (initCode p).drop 3) (linearState base u) =
      some (initial p base u) := by
  simp only [RebasedCumProgram.decodeCode,initCode,List.drop,List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show
    evalOp (.mod .nat .nil (.ref .nat [] "linear") (.constNat (p.S/p.BS))) _ =
      some (Tile.scalar (decoded p base u).feature) from by
        simp only [linearState,evalOp,BlockState.setReg_same,Option.bind_eq_bind,Option.bind_some]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show
    evalOp (.mod .nat .nil (.div .nat .nil (.ref .nat [] "linear") (.constNat (p.S/p.BS)))
      (.constNat (p.T/p.BT))) _ = some (Tile.scalar (chunk p base u)) from by
        simp only [linearState,evalOp,BlockState.setReg_same,BlockState.setReg_ne_name,
          Option.bind_eq_bind,Option.bind_some]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show
    evalOp (.div .nat .nil (.ref .nat [] "linear") (.constNat ((p.S/p.BS)*(p.T/p.BT)))) _ =
      some (Tile.scalar (decoded p base u).head) from by
        simp only [linearState,evalOp,BlockState.setReg_same,BlockState.setReg_ne_name,
          Option.bind_eq_bind,Option.bind_some]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.arange p.BT) _ = some
    (⟨fun i : TileIndex [p.BT] => i.1.val⟩ : Tile .nat [p.BT]) from by rw [evalOp]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ChunkCumsumVector.RealInit.triangular_eval p _ (by simp))),stepStmts.nil]
  rfl

def q (p : Layout) (base : Nat) (u : BlockState) :=
  RebasedCumAccess.tileLayout p (decoded p base u) (chunk p base u)

def result (p : Layout) (s o : RegionName) (base : Nat) (u : BlockState) :=
  RealCumExecution.result (q p base u) RebasedCumAccess.localLaunch s o
    (u.readMem s) (u.undef s) 0 (initial p base u)

theorem active_executes (p : Layout) (s o : RegionName) (base : Nat) (swapped : Bool) (u : BlockState) :
    stepStmts (RebasedCumProgram.activeCode p s o swapped) (linearState base u) =
      some (result p s o base u) := by
  rw [RebasedCumProgram.activeCode,stepStmts.append_some (initializes p base u)]
  exact RebasedCumAccess.computes p (decoded p base u) s o (u.readMem s) (u.undef s)
    (chunk p base u) swapped (initial p base u) (fun _ => rfl)
    (by simp [initial]) (by simp [initial]) (by simp [initial]) (fun _ => rfl) (by simp [initial])

def enabled (p : Layout) (heads base : Nat) (u : BlockState) :=
  linear base u < heads*(p.T/p.BT)*(p.S/p.BS)
instance (p : Layout) (heads base : Nat) (u : BlockState) : Decidable (enabled p heads base u) :=
  inferInstanceAs (Decidable (_ < _))

def footprint (p : Layout) (heads base : Nat) (o : RegionName) (u : BlockState) : WriteFootprint :=
  fun cell => enabled p heads base u ∧
    RealCumExecution.footprint (q p base u) RebasedCumAccess.localLaunch 0 o cell

theorem test_eval (p : Layout) (heads base : Nat) (u : BlockState) :
    evalOp (.lt .nat .nil (.ref .nat [] "linear") (.constNat (heads*(p.T/p.BT)*(p.S/p.BS))))
      (linearState base u) = some (Tile.scalar (decide (enabled p heads base u))) := by
  simp only [evalOp,linearState,BlockState.setReg_same,Option.bind_eq_bind,Option.bind_some]
  rfl

theorem executes (p : Layout) (s o : RegionName) (heads base : Nat) (swapped : Bool)
    (dtype out_dtype : FloatDType) (u : BlockState)
    (hinj : enabled p heads base u →
      RealCumExecution.NoCollision (q p base u) RebasedCumAccess.localLaunch 0) :
    ∃ v, exec (RebasedCumProgram.source p s o heads base swapped dtype out_dtype).eraseDType.toAlgKernel u = some v ∧
      (enabled p heads base u → ∀ i : TileIndex [p.BT,p.BS],
        v.readMem o (RealCumExecution.offset (q p base u) RebasedCumAccess.localLaunch 0 i) =
          RealCumExecution.expected (q p base u) RebasedCumAccess.localLaunch 0 (u.readMem s) i) ∧
      BlockState.WriteWithin (footprint p heads base o u) u v := by
  have hlin : evalOp (.add .nat .nil (.programId 0) (.constNat base)) u =
      some (Tile.scalar (linear base u)) := by
    simp only [evalOp_add,evalOp_programId,evalOp_constNat]; rfl
  by_cases ha : enabled p heads base u
  · refine ⟨result p s o base u,?_,?_,?_⟩
    · rw [exec,RebasedCumProgram.source_body,RebasedCumProgram.body,
        stepStmts.cons_some (stepStmt_assign_eq_some hlin)]
      change stepStmts [.ifThen _ _] (linearState base u) = _
      simp only [stepStmts,stepStmt,test_eval,Option.bind_eq_bind,Option.bind_some,
        Tile.scalar,decide_eq_true ha,↓reduceIte,active_executes]
    · intro _ i
      exact RebasedCumAccess.readback p (decoded p base u) (chunk p base u) s o
        (u.readMem s) (u.undef s) (initial p base u) (hinj ha) i
    · intro region off hn
      exact RebasedCumAccess.frame p (decoded p base u) (chunk p base u) s o
        (u.readMem s) (u.undef s) (initial p base u) region off (fun h => hn ⟨ha,h⟩)
  · refine ⟨linearState base u,?_,fun h => False.elim (ha h),?_⟩
    · rw [exec,RebasedCumProgram.source_body,RebasedCumProgram.body,
        stepStmts.cons_some (stepStmt_assign_eq_some hlin)]
      change stepStmts [.ifThen _ _] (linearState base u) = _
      simp only [stepStmts,stepStmt,test_eval,Option.bind_eq_bind,Option.bind_some,
        Tile.scalar,decide_eq_false ha,Bool.false_eq_true,↓reduceIte]
    · intro _ _ _; rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumExecution
