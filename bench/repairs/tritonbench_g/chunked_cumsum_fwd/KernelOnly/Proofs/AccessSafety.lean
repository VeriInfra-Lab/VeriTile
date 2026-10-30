import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelGrid
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
open VeriTile.Triton FP.Structural Math
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Check the actual quoted source, including uint64 index casts, optional
bias/softplus branches, both stores and the runtime clamp registers. -/
theorem source_traceSafe {α : Type} [Inhabited α]
    (c : Geometry) (p : Primitives α c) (sizes : Sizes c)
    (hasBias useSoftplus : Bool) (lower upper : α)
    (l : Launch) (hl : c.valid l) (s : State α) (bounds : RegionName → Nat)
    (hdt : ∀ (i : Fin c.BH) (j : Fin c.BC), activeInput c.layout l i.val j.val →
      dtAddress c.layout l i.val j.val < bounds "dt")
    (hbias : hasBias = Bool.true → ∀ i : Fin c.BH, head c.layout l i.val < c.heads →
      head c.layout l i.val * c.biasHead < bounds "bias")
    (hrate : ∀ i : Fin c.BH, head c.layout l i.val < c.heads →
      head c.layout l i.val * c.rateHead < bounds "rate")
    (hout : ∀ (i : Fin c.BH) (j : Fin c.BC), activeOutput c.layout l i.val j.val →
      outAddress c.layout l i.val j.val < bounds "out")
    (hsum : ∀ (i : Fin c.BH) (j : Fin c.BC), activeOutput c.layout l i.val j.val →
      outAddress c.layout l i.val j.val < bounds "sum") :
    FP.Structural.TraceSafe (engine p.model) bounds (kernel c hasBias useSoftplus)
      (tileState l (bindBounds s lower upper)) := by
  obtain ⟨hn, hh, hc, hb, hk, hhead⟩ := scalar_mods c sizes l hl
  simp only [Nat.reducePow] at hn hh hc hb hk hhead
  have hrow (i : Fin c.BH) : i.val % 18446744073709551616 = i.val := row_mod c i
  have hcol (j : Fin c.BC) : j.val % 18446744073709551616 = j.val := column_mod c sizes j
  cases hasBias <;> cases useSoftplus
  all_goals
    simp (maxSteps := 1500000) [FP.Structural.TraceSafe, kernel, source,
      ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
      Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
      Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
      Safety.maskActive_unfold, run, step, FP.Structural.evalExpr, evalComputeOp,
      evalOp_unfold, store, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, ComputeDType.eraseDType, numeric, bop, natLt, natLe,
      numericLt, numericLe, tileState, bindBounds, ofFloat, toFloat, integralMod,
      IntegerExpr.same_left, IntegerExpr.same_right, Op.natToUnsigned,
      TileShape.axisDim, Region.cast, State.setReg, Geometry.layout, scatter_regs,
      p.le_support, p.minimum_support, p.scan_support, hn, hh, hc, hb, hk, hhead,
      hrow, hcol, limit_eq, Nat.add_assoc]
  all_goals
    simp only [activeInput, activeOutput, dtAddress, outAddress, head, Geometry.layout,
      Nat.add_assoc] at hdt hbias hrate hout hsum
  all_goals aesop

#axiomsClean source_traceSafe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
