import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerSafety

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerGrid
open VeriTile.Triton IntegerSteps
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

private theorem int_read_congr {s t : BlockState} {r : RegionName} {o : Nat}
    (h : s.mem r o = t.mem r o) : s.readMemValue .int r o = t.readMemValue .int r o := by
  simp only [BlockState.readMemValue, BlockState.readMemTyped, h]

private theorem real_read_congr {s t : BlockState} {r : RegionName} {o : Nat}
    (h : s.mem r o = t.mem r o) : s.readMem r o = t.readMem r o := by
  simp only [BlockState.readMem, h]

private theorem gsTask_ne_of_ne {tile_size0 : Nat} (P C : Nat) (hC : 0 < C)
    {j j' : Nat} (hne : j ≠ j') (i i' : Fin tile_size0) :
    taskIndex (P + j * C) tile_size0 i ≠ taskIndex (P + j' * C) tile_size0 i' := by
  have key : ∀ a b : Nat, a < b → ∀ x y : Fin tile_size0,
      taskIndex (P + a * C) tile_size0 x < taskIndex (P + b * C) tile_size0 y := by
    intro a b hab x y
    have h1 : a * C + 1 ≤ b * C := by
      have h := Nat.mul_le_mul_right C (Nat.succ_le_of_lt hab)
      calc a * C + 1 ≤ a * C + C := by omega
        _ = (a + 1) * C := by ring
        _ ≤ b * C := h
    have h2 : (P + a * C + 1) * tile_size0 ≤ (P + b * C) * tile_size0 :=
      Nat.mul_le_mul_right _ (by omega)
    have hx := x.isLt
    calc taskIndex (P + a * C) tile_size0 x
        = (P + a * C) * tile_size0 + x.val := rfl
      _ < (P + a * C + 1) * tile_size0 := by
          have hexp : (P + a * C + 1) * tile_size0
              = (P + a * C) * tile_size0 + tile_size0 := by ring
          omega
      _ ≤ (P + b * C) * tile_size0 := h2
      _ ≤ taskIndex (P + b * C) tile_size0 y := Nat.le_add_right _ _
  rcases Nat.lt_or_ge j j' with h | h
  · exact Nat.ne_of_lt (key j j' h i i')
  · exact (Nat.ne_of_lt (key j' j
      (Nat.lt_of_le_of_ne h (fun e => hne e.symm)) i' i)).symm


/-- The complete dynamic grid loop succeeds and is safe for arbitrary trip
counts. Its invariant retains the initial typed exponent memory, discharges
signed32 checks on every computed lane, and frames all raw memory cells. -/
theorem range_run (candidate : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N B : Nat) (hstride : 0 < sy) (hdisj : X ≠ Y)
    (P C : Nat) (hC : 0 < C) (tiles : Nat) (bounds : RegionBounds)
    (hin : ∀ (j : Fin tiles) (i : Fin B), taskIndex (P+j.val*C) B i < N →
      taskIndex (P+j.val*C) B i * sx < bounds X)
    (hout : ∀ (j : Fin tiles) (i : Fin B), taskIndex (P+j.val*C) B i < N →
      taskIndex (P+j.val*C) B i * sy < bounds Y) :
    ∀ n c : Nat, c+n=tiles → ∀ s : BlockState,
      s.regs .nat [] "pid" = some (Tile.scalar P) →
      s.regs .nat [] "num_ctas" = some (Tile.scalar C) →
      s.undef = (fun _ _ => 0) →
      (∀ (j : Nat) (i : Fin B), c ≤ j → j < tiles → taskIndex (P+j*C) B i < N →
        Int32Bits.inRange (s.readMemValue .int X (taskIndex (P+j*C) B i * sx)) = Bool.true) →
      Stmt.forRangeTraceSafe bounds "j" c tiles 1
        (IntegerSource.loop candidate base X Y sx sy N B) s ∧
      ∃ t, stepForRangeAux "j" c tiles 1
          (IntegerSource.loop candidate base X Y sx sy N B) s = some t ∧
        (∀ (j : Nat) (i : Fin B), c ≤ j → j < tiles → taskIndex (P+j*C) B i < N →
          t.readMem Y (taskIndex (P+j*C) B i * sy) =
            IntegerSteps.powSpec s X sx base (taskIndex (P+j*C) B i)) ∧
        (∀ r o, (r ≠ Y ∨ ∀ (j : Nat) (i : Fin B), c ≤ j → j < tiles →
          taskIndex (P+j*C) B i < N → o ≠ taskIndex (P+j*C) B i * sy) →
          t.mem r o = s.mem r o) := by
  intro n
  induction n with
  | zero =>
    intro c hc s hpid hctas hu hvalid
    have hge : tiles ≤ c := by omega
    refine ⟨?_, s, stepForRangeAux.step_ge one_ne_zero hge, ?_, fun _ _ _ => rfl⟩
    · rw [Stmt.forRangeTraceSafe]
      simp [show ¬c < tiles by omega]
    · intro j i hcj hj hi
      omega
  | succ n ih =>
    intro c hc s hpid hctas hu hvalid
    have hlt : c < tiles := by omega
    let tJ := s.setReg "j" .nat [] (Tile.scalar c)
    let addV : Tile .nat [] := Tile.bop (NumericDType.add .nat) .nil
      (Tile.scalar P) (Tile.bop (NumericDType.mul .nat) .nil (Tile.scalar c) (Tile.scalar C))
    let tA := tJ.setReg "tile_id" .nat [] addV
    have hpidJ : tJ.regs .nat [] "pid" = some (Tile.scalar P) := by simp [tJ,hpid]
    have hctasJ : tJ.regs .nat [] "num_ctas" = some (Tile.scalar C) := by simp [tJ,hctas]
    have hadd : addV.data PUnit.unit = P+c*C := rfl
    have hAssign : stepStmt (.assign .nat [] "tile_id"
        (.add .nat .nil (.ref .nat [] "pid")
          (.mul .nat .nil (.ref .nat [] "j") (.ref .nat [] "num_ctas")))) tJ = some tA :=
      stepStmt_assign_eq_some (by
        simp only [evalOp_add, evalOp_mul, evalOp_ref, hpidJ, hctasJ,
          show tJ.regs .nat [] "j" = some (Tile.scalar c) by simp [tJ], Option.bind_some]
        rfl)
    obtain ⟨tb, hstep, hv, hpOut, hpRegion, hregs, hframe⟩ :=
      IntegerSteps.perTile_steps candidate base X Y sx sy N B hstride (P+c*C)
        tA addV (by simp [tA]) hadd hu (fun i hi => hvalid c i (by omega) hlt hi)
    have hbody : stepStmts (IntegerSource.loop candidate base X Y sx sy N B) tJ = some tb := by
      unfold IntegerSource.loop
      rw [stepStmts.cons_some hAssign]
      exact hstep
    have hbodySafe : Stmt.TraceSafeList bounds
        (IntegerSource.loop candidate base X Y sx sy N B) tJ := by
      unfold IntegerSource.loop
      refine Stmt.TraceSafeList.cons_intro
        (by simp only [Stmt.TraceSafe, Op.SafeAt, and_self]) (fun t ht => ?_)
      rw [hAssign] at ht
      obtain rfl := Option.some.inj ht
      exact IntegerSteps.perTile_traceSafe candidate base X Y sx sy N B bounds
        (P+c*C) tA addV (by simp [tA]) hadd
        (fun i hi => hin ⟨c,hlt⟩ i hi) (fun i hi => hout ⟨c,hlt⟩ i hi)
    have hpidB : tb.regs .nat [] "pid" = some (Tile.scalar P) := by
      rw [hregs _ _ _ (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
      simp [tA,tJ,hpid]
    have hctasB : tb.regs .nat [] "num_ctas" = some (Tile.scalar C) := by
      rw [hregs _ _ _ (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
      simp [tA,tJ,hctas]
    have huB : tb.undef = fun _ _ => 0 :=
      (stepStmts_undef _ tA tb (IntegerSteps.perTile_flat candidate base X Y sx sy N B) hstep).trans hu
    have hinput : ∀ o, tb.readMemValue .int X o = s.readMemValue .int X o := by
      intro o
      exact int_read_congr (hframe X o (Or.inl hdisj))
    have hvalidB : ∀ (j : Nat) (i : Fin B), c+1 ≤ j → j < tiles →
        taskIndex (P+j*C) B i < N →
        Int32Bits.inRange (tb.readMemValue .int X (taskIndex (P+j*C) B i * sx)) = Bool.true := by
      intro j i hcj hj hi
      rw [hinput]
      exact hvalid j i (by omega) hj hi
    obtain ⟨hsRest,t,heRest,hvRest,hfRest⟩ :=
      ih (c+1) (by omega) tb hpidB hctasB huB hvalidB
    refine ⟨?_,t,?_,?_,?_⟩
    · rw [Stmt.forRangeTraceSafe]
      simp only [one_ne_zero, hlt, ↓reduceIte]
      refine ⟨hbodySafe, ?_⟩
      rw [hbody]
      exact hsRest
    · rw [stepForRangeAux.step_lt one_ne_zero hlt, hbody, Option.bind_some]
      exact heRest
    · intro j i hcj hj hi
      rcases Nat.eq_or_lt_of_le hcj with rfl | hgt
      · have hkeep := hfRest Y (taskIndex (P+c*C) B i * sy) (Or.inr
          (fun j' i' hcj' _ _ heq => absurd (Nat.eq_of_mul_eq_mul_right hstride heq)
            (gsTask_ne_of_ne P C hC (show c ≠ j' by omega) i i')))
        rw [real_read_congr hkeep, hv i hi]
        rfl
      · rw [hvRest j i (by omega) hj hi]
        simp only [IntegerSteps.powSpec, hinput]
    · intro r o hcnd
      rw [hfRest r o (hcnd.elim Or.inl (fun hh => Or.inr
        (fun j i hcj hj hi => hh j i (by omega) hj hi))),
        hframe r o (hcnd.elim Or.inl (fun hh => Or.inr
          (fun i hi => hh c i (Nat.le_refl c) hlt hi)))]
      rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerGrid
