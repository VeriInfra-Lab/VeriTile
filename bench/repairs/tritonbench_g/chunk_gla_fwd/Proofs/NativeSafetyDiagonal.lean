import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyBasics

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPDiagonal
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

structure DiagonalCapacity (split : Bool) (c : Layout) (s : State α)
    (bounds : RegionName → Nat) : Prop where
  query : ∀ i : TileIndex [c.BC,c.BK], row split c s i.1.val < c.T →
    feature split c s i.2.1.val < c.K →
    s.pids 2*(c.T*c.K)+row split c s i.1.val*c.K+feature split c s i.2.1.val < bounds "q"
  gateQuery : ∀ i : TileIndex [c.BC,c.BK], row split c s i.1.val < c.T →
    feature split c s i.2.1.val < c.K →
    s.pids 2*(c.T*c.K)+row split c s i.1.val*c.K+feature split c s i.2.1.val < bounds "g"
  key : ∀ j, j < columns split c s → ∀ i : Fin c.BK, feature split c s i.val < c.K →
    keyAddress split c s j i.val < bounds "k"
  gateKey : ∀ j, j < columns split c s → ∀ i : Fin c.BK, feature split c s i.val < c.K →
    keyAddress split c s j i.val < bounds "g"
  output : ∀ j, j < columns split c s → ∀ i : Index c, row split c s i.1.val < c.T →
    outputAddress split c s j i < bounds (out split)

theorem diagonal_entry (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (bounds : RegionName → Nat) (s : State α) (capacity : DiagonalCapacity split c s bounds) :
    Safety.traceSafeList (engine M) bounds
      ((branch split c scale).before ++ (parts split c scale).before) s := by
  cases split
  all_goals
    simp [parts,branch,kernel,TypedIntra.original,TypedIntra.source,TypedSplit.original,TypedSplit.source,
      ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Source.prefixBefore]
    simp only [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
    simp [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  all_goals
    constructor
    · intro i j row feature
      simpa [FPDiagonal.feature] using
        capacity.query (i,j,PUnit.unit) row (by simpa [FPDiagonal.feature] using feature)
    · intro i j row feature
      simpa [FPDiagonal.feature] using
        capacity.gateQuery (i,j,PUnit.unit) row (by simpa [FPDiagonal.feature] using feature)

theorem diagonal_body (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (bounds : RegionName → Nat) (initial current : State α) (j : Nat)
    (hj : j < columns split c initial) (ctx : Context M split c initial current j)
    (capacity : DiagonalCapacity split c initial bounds) :
    Safety.traceSafeList (engine M) bounds (parts split c scale).body
      (current.setReg "j" .nat [] (fun _ => j)) := by
  rw [←List.take_append_drop (if split then 6 else 5) (parts split c scale).body,
    Safety.traceSafeList_append]
  constructor
  swap
  · intro t _
    apply pure_trace
    cases split
    all_goals
      simp [parts,branch,kernel,TypedIntra.original,TypedIntra.source,TypedSplit.original,TypedSplit.source,
        ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Source.prefixBefore]
      simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
      simp [Safety.safeOp_unfold]
  cases split
  all_goals
    simp [parts,branch,kernel,TypedIntra.original,TypedIntra.source,TypedSplit.original,TypedSplit.source,
      ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Source.prefixBefore]
    simp only [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
    simp [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,ctx.key,ctx.gateKey,ctx.featureMask,ctx.rowMask,
      ctx.offsets,ctx.rowIndex,ctx.query,ctx.gate,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  all_goals
    refine ⟨?_,?_,?_⟩
    · intro i active
      exact capacity.key j hj i active
    · intro i active
      exact capacity.gateKey j hj i active
    · intro i unit active
      cases unit
      exact capacity.output j hj (i,PUnit.unit) active

#axiomsClean diagonal_entry
#axiomsClean diagonal_body
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
