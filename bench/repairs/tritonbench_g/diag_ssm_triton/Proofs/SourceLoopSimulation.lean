import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceRealStepRelation
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

/-- A local step relation lifts to the original signed loop and the natural
mathematical loop, including the empty range. -/
theorem paired_loop (idx : RegName) (n : Nat) (left right : List Stmt)
    (P : BlockState → BlockState → Prop)
    (hs : ∀ i a b, i < n → P a b → ∃ va vb,
      stepStmts left (a.setReg idx .int [] (Tile.scalar (i : Int))) = some va ∧
      stepStmts right (b.setReg idx .nat [] (Tile.scalar i)) = some vb ∧ P va vb)
    (i : Nat) (hi : i ≤ n) (a b : BlockState) (h : P a b) :
    ∃ va vb, stepForRangeIntAux idx (i : Int) (n : Int) 1 left a = some va ∧
      stepForLoopAux idx i n right b = some vb ∧ P va vb := by
  by_cases hlt : i < n
  · obtain ⟨aa, bb, ha, hb, hab⟩ := hs i a b hlt h
    obtain ⟨va, vb, hla, hlb, hp⟩ := paired_loop idx n left right P hs (i+1) (by omega) aa bb hab
    refine ⟨va, vb, ?_, ?_, hp⟩
    · rw [stepForRangeIntAux.step_lt (by decide) (by exact_mod_cast hlt), ha]
      simpa only [Option.bind_some, ← Int.natCast_add, Int.natCast_one] using hla
    · rw [stepForLoopAux.step_lt hlt, hb]
      exact hlb
  · have hge : n ≤ i := by omega
    exact ⟨a, b, stepForRangeIntAux.step_ge (by decide) (by exact_mod_cast hge),
      stepForLoopAux.step_ge hge, h⟩
termination_by n-i

/-- Per-iteration safety is transported only at related reachable states. -/
theorem paired_loop_safe (bounds : RegionBounds) (idx : RegName) (n : Nat) (left right : List Stmt)
    (P : BlockState → BlockState → Prop)
    (hs : ∀ i a b, i < n → P a b → ∃ va vb,
      stepStmts left (a.setReg idx .int [] (Tile.scalar (i : Int))) = some va ∧
      stepStmts right (b.setReg idx .nat [] (Tile.scalar i)) = some vb ∧ P va vb)
    (safe : ∀ i a b, i < n → P a b →
      (Stmt.TraceSafeList bounds left (a.setReg idx .int [] (Tile.scalar (i : Int))) ↔
       Stmt.TraceSafeList bounds right (b.setReg idx .nat [] (Tile.scalar i))))
    (i : Nat) (hi : i ≤ n) (a b : BlockState) (h : P a b) :
    Stmt.forRangeIntTraceSafe bounds idx (i : Int) (n : Int) 1 left a ↔
      Stmt.forLoopTraceSafe bounds idx i n right b := by
  rw [Stmt.forRangeIntTraceSafe, Stmt.forLoopTraceSafe]
  by_cases hlt : i < n
  · obtain ⟨aa, bb, ha, hb, hab⟩ := hs i a b hlt h
    simp only [Nat.one_ne_zero, Int.ofNat_lt, hlt, ↓reduceIte, safe i a b hlt h, ha, hb]
    apply and_congr_right
    intro _
    simpa only [Int.natCast_add, Int.natCast_one] using
      paired_loop_safe bounds idx n left right P hs safe (i+1) (by omega) aa bb hab
  · simp only [Nat.one_ne_zero, Int.ofNat_lt, hlt, ↓reduceIte]
termination_by n-i

#axiomsClean paired_loop
#axiomsClean paired_loop_safe
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceReal
