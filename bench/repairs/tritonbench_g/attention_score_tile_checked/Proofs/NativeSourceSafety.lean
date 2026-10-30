import bench.repairs.tritonbench_g.attention_score_tile_checked.Kernels
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.NativeSafetySteps

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety Reused FPEngine FPContext
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Bounds (B D : Nat) (p : RealSource.Parameters) (s : State α)
    (bounds : RegionName → Nat) : Prop where
  key : ∀ (d : Fin D) (j : Fin B), p.evenN=Bool.true ∨ s.pids 0*B+j.val<p.keyContext →
    baseK p s+d.val*p.keyDim+(s.pids 0*B+j.val)*p.keyRow < bounds p.K
  query : ∀ row, row%B=0 → row<p.queryContext → ∀ (i : Fin B) (d : Fin D),
    p.evenM=Bool.true ∨ row+i.val<p.queryContext →
      baseQ p s+(row+i.val)*p.queryRow+d.val*p.queryDim < bounds p.Q
  normalizer : ∀ row, row<p.queryContext → baseM p s+row < bounds p.M
  output : ∀ i : Fin B, s.pids 0*B+i.val<p.keyContext →
    FPOutput.address B p s (i,()) < bounds p.Out

theorem prelude (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (s : State α) (bounds : RegionName → Nat) (hb : Bounds B D p s bounds)
    (keyOffsetBound : s.pids 0*B < 2147483648) :
    traceSafeList (engine (withDot M dot)) bounds (parts B D p).before s := by
  have remainder (a b : Nat) : (a : Int) - ((b*(a/b) : Nat) : Int) = ((a%b : Nat) : Int) := by
    rw [Nat.cast_mul, Int.natCast_ediv, ← Int.emod_def, ← Int.natCast_emod]
  have notBelowProduct (a b : Nat) : ¬ (a : Int) < ((b*(a/b) : Nat) : Int) := by
    have h := Nat.mod_add_div a b
    omega
  have bits : (s.pids 0*B : Int) % 4294967296 = ((s.pids 0*B : Nat) : Int) :=
    Int.emod_eq_of_lt (by positivity) (by exact_mod_cast (show s.pids 0*B < 4294967296 by omega))
  have bitsNat : s.pids 0*B % 4294967296 = s.pids 0*B := Nat.mod_eq_of_lt (by omega)
  have signBit : 2*(s.pids 0*B) < 4294967296 := by omega
  rw [←List.take_append_drop 12 (parts B D p).before,traceSafeList_append]
  refine ⟨?_,fun next _ => prelude_tail M dot B D p next bounds⟩
  by_cases hn : p.evenN=Bool.true
  all_goals
    simp [parts,FPEngine.kernel,RealSource.source,TypedSource.original,Whole002.TypedSource.widened,hn,
      ComputeKernel.surfaceBody,Source.intRange?,Source.prefixBefore,Option.map,Option.getD,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  all_goals try simp only [safeBlockPtr2,safeAdvance2]
  all_goals simp [safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,BlockPtr.inBounds_2d_offsets,
      BlockPtr.address_2d_offsets,TileShape.indexToList]
  all_goals try norm_num only [Int.natCast_nonneg, not_lt_of_ge, ↓reduceIte]
  all_goals simp only [ite_false, ite_true, notBelowProduct, remainder,
    ← Int.natCast_ediv, ← Int.natCast_emod, ← Int.natCast_mul, ← Int.natCast_add,
    Int.toNat_natCast]
  all_goals try norm_num
  all_goals rw [bits]
  all_goals simp only [Int.toNat_natCast, bitsNat, signBit, ite_true,
    max_eq_left (Int.natCast_nonneg _)]
  all_goals rw [Int.emod_eq_of_lt (Int.natCast_nonneg _) (by
    exact_mod_cast (show s.pids 0*B < 4294967296 by omega))]
  all_goals simp only [Int.toNat_natCast]
  · intro d j
    simpa only [baseK] using hb.key d j (Or.inl hn)
  · intro d j hj
    simpa only [baseK] using hb.key d j (Or.inr hj)

theorem body (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (initial current : State α) (row : Nat) (acc : Fin B → α)
    (ctx : Context M B D p initial row acc current) (bounds : RegionName → Nat)
    (hb : Bounds B D p initial bounds) (hrow : row%B=0) (hstop : row<p.queryContext) :
    traceSafeList (engine (withDot M dot)) bounds (parts B D p).body
      (current.setReg "start_m" .int [] (fun _ => (row : Int))) := by
  rw [←List.take_append_drop 3 (parts B D p).body,traceSafeList_append]
  refine ⟨?_,fun next _ => body_tail M dot B D p next bounds⟩
  by_cases hm : p.evenM=Bool.true
  all_goals
    simp [parts,FPEngine.kernel,RealSource.source,TypedSource.original,Whole002.TypedSource.widened,hm,
      ComputeKernel.surfaceBody,Source.intRange?,Source.prefixBefore,Option.map,Option.getD,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  all_goals try simp only [safeBlockPtr2,safeAdvance2]
  all_goals simp [safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,queryPointer,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      ctx.memory,ctx.startN,ctx.query,ctx.normalizer,ctx.key,ctx.scale,ctx.output]
  all_goals simp only [← Int.natCast_add, Int.ofNat_lt]
  all_goals
    constructor
    · intro i d
      first
      | exact hb.query row hrow hstop i d (Or.inl hm)
      | intro hi; exact hb.query row hrow hstop i d (Or.inr hi)
    · intro i hi
      simpa [Nat.add_assoc] using hb.normalizer (row+i.val) hi

theorem output (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (initial current : State α) (row : Nat) (acc : Fin B → α)
    (ctx : Context M B D p initial row acc current) (bounds : RegionName → Nat)
    (hb : Bounds B D p initial bounds) :
    traceSafeList (engine (withDot M dot)) bounds (parts B D p).after current := by
  simp [parts,FPEngine.kernel,RealSource.source,TypedSource.original,Whole002.TypedSource.widened,
    ComputeKernel.surfaceBody,Source.intRange?,Source.prefixBefore,Option.map,Option.getD,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeBlockPtr2,safeAdvance2,safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.startN,ctx.batch,ctx.head,ctx.output,FP.Structural.store]
  simp only [← Int.natCast_add, ← Int.natCast_mul, ← Int.natCast_ediv,
    ← Int.natCast_emod, Int.toNat_natCast, Int.ofNat_lt]
  intro i hi
  simpa [FPOutput.address,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using
    hb.output i (by simpa [Nat.add_comm] using hi)

end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeSafety
