import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOSafetyPrelude

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
open VeriTile Triton FP.Structural Quote
noncomputable section
variable {α : Type} [Inhabited α]

theorem traceSafe_invariant (M : Algebra α) (arithmetic : IntegerDot.Arithmetic)
    (context : IntegerDot.Context) (bounds : RegionName → Nat) (code : List ComputeStmt)
    (Inv : State α → Prop)
    (safe : ∀ st∈code,∀ state,Inv state → IntegerDot.stepSafe M arithmetic context bounds st state)
    (keep : ∀ st∈code,∀ state next,Inv state → IntegerDot.step M arithmetic context st state=some next → Inv next)
    (state : State α) (h : Inv state) : IntegerDot.traceSafeList M arithmetic context bounds code state := by
  induction code generalizing state with
  | nil => exact IntegerDot.traceSafeList_nil _ _ _ _ _
  | cons st rest ih =>
    apply (IntegerDot.traceSafeList_cons M arithmetic context bounds st rest state).mpr
    refine ⟨safe st (by simp) state h,?_⟩
    intro next executed
    exact ih (fun a ha => safe a (by simp [ha])) (fun a ha => keep a (by simp [ha]))
      next (keep st (by simp) state next h executed)

theorem assigned_keeps_reg {dtype shape name} {values : Option (Values α dtype shape)}
    {state next : State α} (other : RegName) (d : TileDType) (sh : TileShape) (different : other≠name)
    (executed : (values.bind (fun value => some (state.setReg name dtype shape value)))=some next) :
    next.regs d sh other=state.regs d sh other := by
  cases hv : values with
  | none => simp [hv] at executed
  | some value =>
    simp only [hv,Option.bind_some,Option.some.injEq] at executed
    subst next
    simp [State.setReg,different]

end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
