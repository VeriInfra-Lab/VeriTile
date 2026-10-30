import VeriTile.Triton.Memory.TraceRewrite

/-! Local transport for attention's single immutable runtime scale register.
The relation preserves complete actual successor states and source access
safety. It does not replace the source executor or assume successful execution. -/
namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealScalarInvariant
open VeriTile Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def Bound (scale : ℝ) (s : BlockState) : Prop :=
  s.regs .real [] "sm_scale" = some (Tile.scalar (some scale : WithBot ℝ))

def bind (s : BlockState) (scale : ℝ) : BlockState :=
  s.setReg "sm_scale" .real [] (Tile.scalar (some scale : WithBot ℝ))

@[simp] theorem bound_bind (s : BlockState) (scale : ℝ) : Bound scale (bind s scale) := by
  simp [Bound,bind]

theorem bound_setReg {scale : ℝ} {s : BlockState} (bound : Bound scale s)
    (name : RegName) (different : name ≠ "sm_scale") (dtype : TileDType)
    (shape : TileShape) (value : Tile dtype shape) :
    Bound scale (s.setReg name dtype shape value) := by
  simpa only [Bound,BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (Ne.symm different)] using bound

structure Clause (scale : ℝ) (left right : Stmt) : Prop where
  execution : ∀ s, Bound scale s → stepStmt left s = stepStmt right s
  preserved : ∀ s t, Bound scale s → stepStmt right s = some t → Bound scale t
  safety : ∀ bounds s, Bound scale s → (left.TraceSafe bounds s ↔ right.TraceSafe bounds s)

structure Block (scale : ℝ) (left right : List Stmt) : Prop where
  execution : ∀ s, Bound scale s → stepStmts left s = stepStmts right s
  preserved : ∀ s t, Bound scale s → stepStmts right s = some t → Bound scale t
  safety : ∀ bounds s, Bound scale s →
    (Stmt.TraceSafeList bounds left s ↔ Stmt.TraceSafeList bounds right s)

namespace Block

theorem nil (scale : ℝ) : Block scale [] [] where
  execution := by intros; rfl
  preserved := by
    intro s t bound ran
    rw [stepStmts] at ran
    have same : s=t := Option.some.inj ran
    subst t
    exact bound
  safety := by intros; rfl

theorem cons {scale : ℝ} {left right : Stmt} {lefts rights : List Stmt}
    (head : Clause scale left right) (tail : Block scale lefts rights) :
    Block scale (left::lefts) (right::rights) where
  execution := by
    intro s bound
    simp only [stepStmts,head.execution s bound]
    cases ran : stepStmt right s with
    | none => rfl
    | some t => exact tail.execution t (head.preserved s t bound ran)
  preserved := by
    intro s t bound ran
    obtain ⟨u,first,rest⟩ := TraceProofs.cons_inv ran
    exact tail.preserved u t (head.preserved s u bound first) rest
  safety := by
    intro bounds s bound
    rw [Stmt.TraceSafeList,Stmt.TraceSafeList,head.safety bounds s bound,head.execution s bound]
    cases ran : stepStmt right s with
    | none => rfl
    | some t => exact and_congr_right (fun _ => tail.safety bounds t (head.preserved s t bound ran))

end Block

namespace Clause

theorem assign (scale : ℝ) (dtype : TileDType) (shape : TileShape)
    (name : RegName) (value : Op dtype shape) (different : name ≠ "sm_scale") :
    Clause scale (.assign dtype shape name value) (.assign dtype shape name value) where
  execution := by intros; rfl
  preserved := by
    intro s t bound ran
    obtain ⟨v,_,rfl⟩ := TraceProofs.assign_inv ran
    exact bound_setReg bound name different dtype shape v
  safety := by intros; rfl

/-- Only the source's runtime scalar read is replaced in this comparison.
Both programs still execute the actual multiplication in the same state. -/
theorem scale (scale : ℝ) (rows cols : Nat) :
    Clause scale
      (.assign .real [rows,cols] "qk" (.mul .real .scalarR
        (.ref .real [rows,cols] "qk") (.ref .real [] "sm_scale")))
      (.assign .real [rows,cols] "qk" (.mul .real .scalarR
        (.ref .real [rows,cols] "qk") (.const scale))) where
  execution := by
    intro s bound
    simp only [Bound] at bound
    simp only [stepStmt,evalOp_mul,evalOp_ref,bound,evalOp_const]
  preserved := (assign scale .real [rows,cols] "qk" _ (by decide)).preserved
  safety := by
    intro bounds s _
    simp [Stmt.TraceSafe,Op.SafeAt.eq_def]

/-- The actual final pointer store keeps every register, independent of its
mask and of the values stored. -/
theorem ptrStore (scale : ℝ) (dtype : TileDType) (shape : TileShape)
    (ptr : Op .ptr shape) (value : Op dtype shape) (mask : Op .bool shape) :
    Clause scale (.store dtype shape (.ptr ptr) value (.mask mask))
      (.store dtype shape (.ptr ptr) value (.mask mask)) where
  execution := by intros; rfl
  preserved := by
    intro s t bound ran
    simp only [stepStmt] at ran
    cases hv : evalOp value s with
    | none => simp [hv] at ran
    | some values =>
      cases hm : evalOp mask s with
      | none => simp [hv,hm] at ran
      | some masks =>
        cases hp : evalOp ptr s with
        | none => simp [hv,hm,hp] at ran
        | some pointers =>
          simp only [hv,hm,hp,Option.map_some,Bind.bind,Option.bind,Option.some.injEq] at ran
          subst t
          simpa only [Bound,BlockState.foldl_writeMemTypedAt_masked_regs] using bound
  safety := by intros; rfl

theorem conditional {scale : ℝ} (condition : Bool) {left right : List Stmt}
    (body : Block scale left right) :
    Clause scale (.ifThen (.constBool condition) left) (.ifThen (.constBool condition) right) where
  execution := by
    intro s bound
    cases condition <;> simp only [stepStmt,evalOp,Bind.bind,Option.bind,Tile.scalar_data,
      Bool.false_eq_true,ite_false,ite_true]
    exact body.execution s bound
  preserved := by
    intro s t bound ran
    cases condition <;> simp only [stepStmt,evalOp,Bind.bind,Option.bind,Tile.scalar_data,
      Bool.false_eq_true,ite_false,ite_true] at ran
    · cases ran; exact bound
    · exact body.preserved s t bound ran
  safety := by
    intro bounds s bound
    cases condition <;> simp only [Stmt.TraceSafe,Op.SafeAt.eq_def,evalOp,
      Tile.scalar_data,Bool.false_eq_true,ite_false,ite_true,true_and]
    exact body.safety bounds s bound

end Clause

private theorem range_execution {scale : ℝ} {left right : List Stmt}
    (body : Block scale left right) (index : RegName) (different : index ≠ "sm_scale")
    (cur stop stride : Nat) (s : BlockState) (bound : Bound scale s) :
    stepForRangeAux index cur stop stride left s = stepForRangeAux index cur stop stride right s := by
  by_cases zero : stride=0
  · subst stride
    simp only [stepForRangeAux.step_zero_step]
  · by_cases active : cur<stop
    · rw [stepForRangeAux.step_lt zero active,stepForRangeAux.step_lt zero active]
      have entered := bound_setReg bound index different .nat [] (Tile.scalar cur)
      rw [body.execution _ entered]
      cases ran : stepStmts right (s.setReg index .nat [] (Tile.scalar cur)) with
      | none => rfl
      | some t =>
        exact range_execution body index different (cur+stride) stop stride t
          (body.preserved _ t entered ran)
    · rw [stepForRangeAux.step_ge zero (by omega),stepForRangeAux.step_ge zero (by omega)]
termination_by stop-cur
decreasing_by omega

private theorem range_preserved {scale : ℝ} {left right : List Stmt}
    (body : Block scale left right) (index : RegName) (different : index ≠ "sm_scale")
    (cur stop stride : Nat) (s t : BlockState) (bound : Bound scale s)
    (executed : stepForRangeAux index cur stop stride right s=some t) : Bound scale t := by
  by_cases zero : stride=0
  · subst stride
    rw [stepForRangeAux.step_zero_step] at executed
    cases executed
    exact bound
  · by_cases active : cur<stop
    · rw [stepForRangeAux.step_lt zero active] at executed
      have entered := bound_setReg bound index different .nat [] (Tile.scalar cur)
      cases ran : stepStmts right (s.setReg index .nat [] (Tile.scalar cur)) with
      | none => simp [ran] at executed
      | some u =>
        exact range_preserved body index different (cur+stride) stop stride u t
          (body.preserved _ u entered ran) (by simpa only [ran,Option.bind_some] using executed)
    · rw [stepForRangeAux.step_ge zero (by omega)] at executed
      cases executed
      exact bound
termination_by stop-cur
decreasing_by omega

private theorem range_safety {scale : ℝ} {left right : List Stmt}
    (body : Block scale left right) (index : RegName) (different : index ≠ "sm_scale")
    (bounds : RegionBounds) (cur stop stride : Nat) (s : BlockState) (bound : Bound scale s) :
    Stmt.forRangeTraceSafe bounds index cur stop stride left s ↔
      Stmt.forRangeTraceSafe bounds index cur stop stride right s := by
  rw [Stmt.forRangeTraceSafe,Stmt.forRangeTraceSafe]
  by_cases zero : stride=0
  · simp only [zero,ite_true]
  · simp only [zero,ite_false]
    by_cases active : cur<stop
    · simp only [active,ite_true]
      have entered := bound_setReg bound index different .nat [] (Tile.scalar cur)
      rw [body.safety bounds _ entered,body.execution _ entered]
      cases ran : stepStmts right (s.setReg index .nat [] (Tile.scalar cur)) with
      | none => rfl
      | some t => exact and_congr_right (fun _ => range_safety body index different bounds
          (cur+stride) stop stride t (body.preserved _ t entered ran))
    · simp only [active,ite_false]
termination_by stop-cur
decreasing_by omega

namespace Clause

theorem dynamicRange {scale : ℝ} (index : RegName) (different : index ≠ "sm_scale")
    (start stop stride : Op .nat []) {left right : List Stmt} (body : Block scale left right) :
    Clause scale (.forRangeDyn index start stop stride left) (.forRangeDyn index start stop stride right) where
  execution := by
    intro s bound
    simp only [stepForRangeAux.forRangeDyn_unfold]
    congr 1
    funext a
    congr 1
    funext b
    congr 1
    funext d
    exact range_execution body index different _ _ _ s bound
  preserved := by
    intro s t bound ran
    simp only [stepForRangeAux.forRangeDyn_unfold] at ran
    cases ha : evalOp start s with
    | none => simp [ha] at ran
    | some a =>
      cases hb : evalOp stop s with
      | none => simp [ha,hb] at ran
      | some b =>
        cases hd : evalOp stride s with
        | none => simp [ha,hb,hd] at ran
        | some d =>
          exact range_preserved body index different _ _ _ s t bound
            (by simpa only [ha,hb,hd,Option.bind_some] using ran)
  safety := by
    intro bounds s bound
    simp only [Stmt.TraceSafe]
    cases evalOp start s <;> cases evalOp stop s <;> cases evalOp stride s <;>
      simp only [range_safety body index different bounds _ _ _ s bound]

end Clause
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealScalarInvariant
