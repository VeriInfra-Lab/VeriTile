import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOSafetyCommon
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOSafetyBounds
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IntegerDotBridge

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
open VeriTile Triton FP.Structural Quote
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def factors (s : Shape) (c : Config) (hasBias : Bool) : List ComputeStmt :=
  (prelude s c hasBias).drop 27

theorem factors_safe (M : Algebra α) (arithmetic : IntegerDot.Arithmetic)
    (context : IntegerDot.Context) (s : Shape) (c : Config) (hasBias : Bool)
    (state : State α) (bounds : RegionName → Nat)
    (ram : Values α .int [c.BM]) (rbn : Values α .int [c.BN])
    (hram : state.regs .int [c.BM] "ram"=some ram)
    (hrbn : state.regs .int [c.BN] "rbn"=some rbn)
    (hx : ∀ i,(ram i).toNat<bounds "state_x")
    (hw : ∀ i,(rbn i).toNat<bounds "state_w") :
    IntegerDot.traceSafeList M arithmetic context bounds (factors s c hasBias) state := by
  apply traceSafe_invariant M arithmetic context bounds (factors s c hasBias)
    (fun state => state.regs .int [c.BM] "ram"=some ram ∧ state.regs .int [c.BN] "rbn"=some rbn)
  · intro st member current hregs
    simp [factors,prelude,source,TypedSource.kernel,TypedSource.skeleton,TypedSource.bindRuntime,
      ComputeKernel.surfaceBody] at member
    rcases member with rfl|rfl|rfl
    all_goals
      simp only [IntegerDot.stepSafe_unfold,Safety.stepSafe_unfold,
        Safety.safeExpr_unfold,Safety.safeCompute_unfold]
      simp [Safety.safeOp_unfold,Safety.addressSafe,Safety.maskActive,evalOp_unfold,
        hregs.1,hregs.2,hram,hrbn,hx,hw,bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals first | exact fun a b => hw (a,b) | exact fun a b => hx (a,b)
  · intro st member current next hregs executed
    simp [factors,prelude,source,TypedSource.kernel,TypedSource.skeleton,TypedSource.bindRuntime,
      ComputeKernel.surfaceBody] at member
    rcases member with rfl|rfl|rfl
    all_goals
      simp only [IntegerDot.step,IntegerDot.arithmeticStep,FP.Structural.step] at executed
      constructor
      · refine (assigned_keeps_reg "ram" .int [c.BM] ?_ executed).trans hregs.1
        decide
      · refine (assigned_keeps_reg "rbn" .int [c.BN] ?_ executed).trans hregs.2
        decide
  · exact ⟨hram,hrbn⟩

theorem prefix_regs (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State α) :
    let state := (preFactorsExecution M q s c hasBias initial).val
    (state.regs .int [c.BM] "ram"=some (register state .int [c.BM] "ram")) ∧
    (state.regs .int [c.BN] "rbn"=some (register state .int [c.BN] "rbn")) := by
  simp [register,preFactorsExecution]

theorem prelude_safe (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State α)
    (valid : initial.pids 0<IntScaledMatmul.Revision001.Geometry.count (geometry s c))
    (bounds : RegionName → Nat) (hb : Bounds s hasBias bounds) :
    IntegerDot.traceSafeList M (SharedInteger.arithmetic q) (SharedInteger.context q s.M s.N)
      bounds (prelude s c hasBias) initial := by
  rw [←List.take_append_drop 27 (prelude s c hasBias),IntegerDot.traceSafeList_append]
  refine ⟨preFactors_safe M _ _ s c hasBias initial bounds,?_⟩
  intro next executed
  rw [SharedInteger.run_eq] at executed
  change Quote.run M q s.M s.N (preFactors s c hasBias) initial=some next at executed
  rw [(preFactorsExecution M q s c hasBias initial).property] at executed
  cases Option.some.inj executed
  have regs := prefix_regs M q s c hasBias initial
  apply factors_safe M _ _ s c hasBias _ bounds _ _ regs.1 regs.2
  · rw [prefix_ram M q s c hasBias initial valid]
    intro i
    exact hb.x _ (Nat.mod_lt _ (IOGeometry.rows_pos s c _ valid))
  · rw [prefix_rbn M q s c hasBias initial valid]
    intro i
    exact hb.w _ (Nat.mod_lt _ (IOGeometry.columns_pos s c _ valid))

#axiomsClean factors_safe
#axiomsClean prefix_regs
#axiomsClean prelude_safe
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
