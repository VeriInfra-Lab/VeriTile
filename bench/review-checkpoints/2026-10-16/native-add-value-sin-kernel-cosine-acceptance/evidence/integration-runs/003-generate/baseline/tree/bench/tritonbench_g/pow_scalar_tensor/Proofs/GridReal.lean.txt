import bench.tritonbench_g.pow_scalar_tensor.Proofs.LegacyReal
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.GridReal
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

private theorem numTiles0_step (s0 tile_size0 : Nat) (t : BlockState) :
    stepStmt (numTiles0Stmt s0 tile_size0) t =
      some (t.setReg "num_tiles0" .nat [] (numTiles0Val s0 tile_size0)) := by
  exact stepStmt_assign_eq_some (by
    simp only [evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat,
      Option.bind_some, numTiles0Val]
    rfl)

private theorem stepStmts_singleton (st : Stmt) (s : BlockState) :
    stepStmts [st] s = stepStmt st s := by
  simp only [stepStmts]
  cases stepStmt st s <;> rfl

set_option maxHeartbeats 1000000 in
private theorem gs_loop_cells (val0 : ℝ)
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 tile_size0 : Nat)
    (hStride : 0 < out0_stride0) (P C : Nat) (tiles_per_cta : Nat) :
    ∀ (n c : Nat), c + n = tiles_per_cta →
    ∀ (t : BlockState),
      t.regs .nat [] "pid" = some (Tile.scalar P) →
      t.regs .nat [] "num_ctas" = some (Tile.scalar C) →
      ∃ t', stepForRangeAux "j" c tiles_per_cta 1
          (Stmt.assign .nat [] "tile_id"
              (Op.add NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
                  (Op.ref .nat [] "num_ctas")))
            :: perTileStmts val0 in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0)
          t = some t'
        ∧ (∀ (r : RegionName) (o : Nat),
            (r ≠ out0_ptr ∨ ∀ (j : Nat) (i : Fin tile_size0), c ≤ j →
              j < tiles_per_cta → taskIndex (P + j * C) tile_size0 i < s0 →
              o ≠ taskIndex (P + j * C) tile_size0 i * out0_stride0) →
            t'.mem r o = t.mem r o) := by
  intro n
  induction n with
  | zero =>
      intro c hc t hpid hctas
      exact ⟨t, stepForRangeAux.step_ge one_ne_zero (by omega), fun r o _ => rfl⟩
  | succ n ih =>
      intro c hc t hpid hctas
      have hlt : c < tiles_per_cta := by omega
      set tJ := t.setReg "j" .nat [] (Tile.scalar c) with htJ
      set addV : Tile .nat [] := Tile.bop (NumericDType.add .nat) Broadcast.nil
        (Tile.scalar P) (Tile.bop (NumericDType.mul .nat) Broadcast.nil
          (Tile.scalar c) (Tile.scalar C)) with haddV
      have hpidJ : tJ.regs .nat [] "pid" = some (Tile.scalar P) := by
        rw [htJ, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hpid
      have hctasJ : tJ.regs .nat [] "num_ctas" = some (Tile.scalar C) := by
        rw [htJ, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        exact hctas
      have haddData : addV.data PUnit.unit = P + c * C := by rw [haddV]; rfl
      have hAssign : stepStmt (Stmt.assign .nat [] "tile_id"
          (Op.add NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid")
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
              (Op.ref .nat [] "num_ctas")))) tJ
          = some (tJ.setReg "tile_id" .nat [] addV) :=
        stepStmt_assign_eq_some (by
          simp only [evalOp_add, evalOp_mul, evalOp_ref, hpidJ, hctasJ,
            show tJ.regs .nat [] "j" = some (Tile.scalar c) by simp [htJ],
            Option.bind_some, haddV]
          rfl)
      obtain ⟨tb, hstepBody, -, -, -, hregs, hcell⟩ :=
        perTile_steps val0 in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0
          hStride (P + c * C) (tJ.setReg "tile_id" .nat [] addV) addV
          (by simp) haddData
      have hbody : stepStmts
          (Stmt.assign .nat [] "tile_id"
              (Op.add NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
                  (Op.ref .nat [] "num_ctas")))
            :: perTileStmts val0 in0_ptr out0_ptr in0_stride0 out0_stride0 s0
              tile_size0) tJ = some tb := by
        rw [stepStmts.cons_some hAssign]; exact hstepBody
      have hpidB : tb.regs .nat [] "pid" = some (Tile.scalar P) := by
        rw [hregs _ _ _ (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        exact hpidJ
      have hctasB : tb.regs .nat [] "num_ctas" = some (Tile.scalar C) := by
        rw [hregs _ _ _ (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        exact hctasJ
      obtain ⟨t', hrest, hcellRest⟩ := ih (c + 1) (by omega) tb hpidB hctasB
      refine ⟨t', ?_, ?_⟩
      · rw [stepForRangeAux.step_lt one_ne_zero hlt, hbody, Option.bind_some]
        exact hrest
      · intro r o hcond
        have hcRest : r ≠ out0_ptr ∨ ∀ (j : Nat) (i : Fin tile_size0), c + 1 ≤ j →
            j < tiles_per_cta → taskIndex (P + j * C) tile_size0 i < s0 →
            o ≠ taskIndex (P + j * C) tile_size0 i * out0_stride0 := by
          rcases hcond with h | h
          · exact Or.inl h
          · exact Or.inr fun j i hcj hj hi => h j i (by omega) hj hi
        have hcHere : r ≠ out0_ptr ∨ ∀ i : Fin tile_size0,
            taskIndex (P + c * C) tile_size0 i < s0 →
            o ≠ taskIndex (P + c * C) tile_size0 i * out0_stride0 := by
          rcases hcond with h | h
          · exact Or.inl h
          · exact Or.inr fun i hi => h c i (Nat.le_refl c) hlt hi
        rw [hcellRest r o hcRest, hcell r o hcHere, htJ]
        simp

set_option maxHeartbeats 1000000 in
/-- Termination and the cell-level frame of the whole grid-stride kernel under
the exact stepper, from an arbitrary launch state. -/
theorem grid_run (val0 : ℝ)
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0) (s₀ : BlockState) :
    ∃ s1, exec ((pow_func_scalar_tensor_kernel_rank_1_grid_stride_surface val0 in0_ptr out0_ptr
        in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
        tile_size0).toAlgKernel) s₀ = some s1
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ out0_ptr ∨ ∀ (j : Nat) (i : Fin tile_size0), j < tiles_per_cta →
            taskIndex (s₀.pids 0 + j * s₀.numPids 0) tile_size0 i < s0 →
            o ≠ taskIndex (s₀.pids 0 + j * s₀.numPids 0) tile_size0 i
                  * out0_stride0) →
          s1.mem r o = s₀.mem r o) := by
  set sp3 := ((s₀.setReg "pid" .nat [] (Tile.scalar (s₀.pids 0))).setReg
      "num_tiles0" .nat [] (numTiles0Val s0 tile_size0)).setReg
      "num_ctas" .nat [] (Tile.scalar (s₀.numPids 0)) with hsp3
  obtain ⟨s1, hloop, hcells⟩ :=
    gs_loop_cells val0 in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0
      hStride (s₀.pids 0) (s₀.numPids 0) tiles_per_cta tiles_per_cta 0 (by omega)
      sp3 (by simp [hsp3]) (by simp [hsp3])
  refine ⟨s1, ?_, ?_⟩
  · rw [show exec ((pow_func_scalar_tensor_kernel_rank_1_grid_stride_surface val0 in0_ptr
          out0_ptr in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
          tile_size0).toAlgKernel) s₀
        = stepStmts ((pow_func_scalar_tensor_kernel_rank_1_grid_stride_surface val0 in0_ptr
          out0_ptr in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
          tile_size0).toAlgKernel).body s₀ from rfl,
      gridStride_body_eq]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s₀))]
    rw [stepStmts.cons_some (numTiles0_step s0 tile_size0 _)]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_numPrograms 0 _))]
    rw [stepStmts_singleton, stepForRangeAux.forRangeDyn_unfold]
    simp only [evalOp_constNat, Option.bind_some]
    exact hloop
  · intro r o hcond
    rw [hcells r o ?_, hsp3]
    · simp
    · rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun j i _ hj hi => h j i hj hi


private theorem range_safety_inv (bounds : RegionBounds) (index : RegName)
    (stop : Nat) (body : List Stmt) (P : Nat → BlockState → Prop)
    (hbody : ∀ c s, c < stop → P c s →
      Stmt.TraceSafeList bounds body (s.setReg index .nat [] (Tile.scalar c)) ∧
      ∃ t, stepStmts body (s.setReg index .nat [] (Tile.scalar c)) = some t ∧ P (c+1) t)
    (cur : Nat) (s : BlockState) (hp : P cur s) :
    Stmt.forRangeTraceSafe bounds index cur stop 1 body s := by
  rw [Stmt.forRangeTraceSafe]
  simp only [one_ne_zero, ↓reduceIte]
  by_cases hc : cur < stop
  · simp only [hc, ↓reduceIte]
    obtain ⟨hsafe, t, he, ht⟩ := hbody cur s hc hp
    refine ⟨hsafe, ?_⟩
    rw [he]
    exact range_safety_inv bounds index stop body P hbody (cur+1) t ht
  · simp only [hc, ↓reduceIte]
termination_by stop - cur
decreasing_by omega

/-- Safety of the actual multi-iteration historical body, with independent
active-load and active-store bounds. This is transported to the typed source
only by the checked whole-source execution/safety relation. -/
theorem grid_safe (val0 : ℝ) (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ (j : Fin tiles_per_cta) (i : Fin tile_size0),
      taskIndex (s.pids 0 + j.val * s.numPids 0) tile_size0 i < s0 →
      taskIndex (s.pids 0 + j.val * s.numPids 0) tile_size0 i * in0_stride0 < bounds in0_ptr)
    (hout : ∀ (j : Fin tiles_per_cta) (i : Fin tile_size0),
      taskIndex (s.pids 0 + j.val * s.numPids 0) tile_size0 i < s0 →
      taskIndex (s.pids 0 + j.val * s.numPids 0) tile_size0 i * out0_stride0 < bounds out0_ptr) :
    (pow_func_scalar_tensor_kernel_rank_1_grid_stride_surface val0 in0_ptr out0_ptr
      in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0).toAlgKernel.TraceSafe bounds s := by
  unfold Kernel.TraceSafe
  rw [gridStride_body_eq]
  refine Stmt.TraceSafeList.cons_intro
    (by simp only [Stmt.TraceSafe, Op.SafeAt]) (fun t1 ht1 => ?_)
  rw [stepStmt_assign_eq_some (evalOp_programId 0 s)] at ht1
  obtain rfl := Option.some.inj ht1
  refine Stmt.TraceSafeList.cons_intro
    (by simp only [numTiles0Stmt, Stmt.TraceSafe, Op.SafeAt, and_self]) (fun t2 ht2 => ?_)
  rw [numTiles0_step s0 tile_size0 _] at ht2
  obtain rfl := Option.some.inj ht2
  refine Stmt.TraceSafeList.cons_intro
    (by simp only [Stmt.TraceSafe, Op.SafeAt]) (fun t3 ht3 => ?_)
  rw [stepStmt_assign_eq_some (evalOp_numPrograms 0 _)] at ht3
  obtain rfl := Option.some.inj ht3
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  simp only [Stmt.TraceSafe, Op.SafeAt, evalOp_constNat, Tile.scalar_data,
    true_and]
  apply range_safety_inv bounds "j" tiles_per_cta _
    (fun _ st => st.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)) ∧
      st.regs .nat [] "num_ctas" = some (Tile.scalar (s.numPids 0)))
  · intro c sc hc hP
    obtain ⟨hpidC, hctasC⟩ := hP
    set scj := sc.setReg "j" .nat [] (Tile.scalar c) with hscj
    have hpidJ : scj.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)) := by
      rw [hscj, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hpidC
    have hctasJ : scj.regs .nat [] "num_ctas" = some (Tile.scalar (s.numPids 0)) := by
      rw [hscj, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hctasC
    set addV : Tile .nat [] := Tile.bop (NumericDType.add .nat) Broadcast.nil
      (Tile.scalar (s.pids 0)) (Tile.bop (NumericDType.mul .nat) Broadcast.nil
        (Tile.scalar c) (Tile.scalar (s.numPids 0))) with haddV
    have hAssign : stepStmt (Stmt.assign .nat [] "tile_id"
        (Op.add .nat .nil (Op.ref .nat [] "pid")
          (Op.mul .nat .nil (Op.ref .nat [] "j") (Op.ref .nat [] "num_ctas")))) scj =
        some (scj.setReg "tile_id" .nat [] addV) :=
      stepStmt_assign_eq_some (by
        simp only [evalOp_add, evalOp_mul, evalOp_ref, hpidJ, hctasJ,
          show scj.regs .nat [] "j" = some (Tile.scalar c) by simp [hscj],
          Option.bind_some, haddV]
        rfl)
    have haddData : addV.data PUnit.unit = s.pids 0 + c * s.numPids 0 := by
      rw [haddV]; rfl
    refine ⟨?_, ?_⟩
    · refine Stmt.TraceSafeList.cons_intro
        (by simp only [Stmt.TraceSafe, Op.SafeAt, and_self]) (fun tb htb => ?_)
      rw [hAssign] at htb
      obtain rfl := Option.some.inj htb
      exact perTile_traceSafe val0 in0_ptr out0_ptr in0_stride0 out0_stride0 s0
        tile_size0 bounds (s.pids 0 + c * s.numPids 0) _ addV
        (BlockState.setReg_same _ _ _ _ _) haddData
        (fun i hi => hin ⟨c, hc⟩ i hi) (fun i hi => hout ⟨c, hc⟩ i hi)
    · obtain ⟨tb, hsteps, -, -, -, hregs, -⟩ := perTile_steps val0 in0_ptr out0_ptr
        in0_stride0 out0_stride0 s0 tile_size0 hStride (s.pids 0 + c * s.numPids 0)
        (scj.setReg "tile_id" .nat [] addV) addV (by simp) haddData
      refine ⟨tb, ?_, ?_, ?_⟩
      · rw [stepStmts.cons_some hAssign]
        exact hsteps
      · rw [hregs _ _ _ (by decide) (by decide) (by decide) (by decide) (by decide)
          (by decide), BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        exact hpidJ
      · rw [hregs _ _ _ (by decide) (by decide) (by decide) (by decide) (by decide)
          (by decide), BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        exact hctasJ
  · exact ⟨by simp, by simp⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.GridReal
