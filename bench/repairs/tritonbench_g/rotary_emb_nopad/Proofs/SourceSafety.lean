import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.TypedDefinitions
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Typed
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Uncached coefficient reads have their own token/pair mask, even when Q
is masked and no K leader writes. In-place payload reads and stores share the
same windows; forming inactive signed pointers does not require backing. -/
theorem uncached_safe {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.UncachedLayout) (s : State A) (bounds : RegionName → Nat)
    (coeff : ∀ sine i, uToken p s i < p.q_total_tokens → i.2.2.1.val < p.HEAD_DIM/2 →
      uCoefficient p s sine i < bounds (if sine then p.sin else p.cos))
    (query : ∀ odd i, uActive p s Bool.false i → uOffset p s Bool.false odd i < bounds p.q)
    (key : s.pids 0 % p.KV_GROUP_NUM = 0 → ∀ odd i, uActive p s Bool.true i →
      uOffset p s Bool.true odd i < bounds p.k) :
    TraceSafe (algebra rest ops) bounds (Kernels.uncached p) s := by
  have headNonneg : ¬(s.pids 0 : Int)<0 := by omega
  have signedLeader :
      ((s.pids 0 : Int)-(p.KV_GROUP_NUM : Int)*((s.pids 0 : Int)/(p.KV_GROUP_NUM : Int))=0) ↔
        s.pids 0%p.KV_GROUP_NUM=0 := by
    rw [←Int.emod_def,←Int.natCast_emod,Int.natCast_eq_zero]
  by_cases leader : s.pids 0 % p.KV_GROUP_NUM = 0
  all_goals
    simp [TraceSafe,Kernels.uncached,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
      safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
      algebra,VeriTile.Bench.Repairs.RotaryEmb.Typed.algebra,numericLt,natLt,numericLe,natLe,numericEq,
      bop,numeric,store,Region.cast,ComputeDType.eraseDType,ofFloat,toFloat,
      integralFloorDiv,integralMod,headNonneg,signedLeader,leader]
    try dsimp only [TileShape.insertAxis,TileShape.dropInsertedIndex]
    all_goals simp only [int_half,←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv,
      Int.toNat_natCast,Int.zero_sub,Int.toNat_neg_natCast,Nat.sub_zero,Int.ofNat_lt]
  all_goals
    repeat' constructor
    all_goals first
      | (intro t d ht hd
         simpa [uCoefficient,uToken] using coeff Bool.false (t,0,d,PUnit.unit) ht hd)
      | (intro t d ht hd
         simpa [uCoefficient,uToken] using coeff Bool.true (t,0,d,PUnit.unit) ht hd)
      | (intro t d head ht hd
         simpa [uOffset,uToken,uHead,uDim] using
           query Bool.false (t,0,d,PUnit.unit) ⟨by simpa using head,ht,hd⟩)
      | (intro t d head ht hd
         simpa [uOffset,uToken,uHead,uDim] using
           query Bool.true (t,0,d,PUnit.unit) ⟨by simpa using head,ht,hd⟩)
      | (intro t d ht hd
         simpa [uOffset,uToken,uHead,uDim] using key leader Bool.false (t,0,d,PUnit.unit) ⟨Or.inl rfl,ht,hd⟩)
      | (intro t d ht hd
         simpa [uOffset,uToken,uHead,uDim] using key leader Bool.true (t,0,d,PUnit.unit) ⟨Or.inl rfl,ht,hd⟩)

/-- Cached source safety includes both signed metadata reads and indirect cache
destinations. The early head return suppresses every access. -/
theorem cached_safe {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.CachedLayout) (s : State A) (bounds : RegionName → Nat)
    (token : s.pids 1 < p.q_total_tokens)
    (length : s.pids 0 < p.Q_HEAD_NUM → s.pids 1 < bounds p.context_lengths)
    (table : s.pids 0 < p.Q_HEAD_NUM →
      s.pids 1*p.bts_stride+((cPast p s).tdiv p.block_size*p.btb_stride).toNat -
      (-((cPast p s).tdiv p.block_size*p.btb_stride)).toNat < bounds p.BLOCK_TABLES)
    (coeff : s.pids 0 < p.Q_HEAD_NUM → ∀ sine i, i.1.val < p.HEAD_DIM/2 →
      cCoefficient p s sine i < bounds (if sine then p.sin else p.cos))
    (input : s.pids 0 < p.Q_HEAD_NUM → ∀ key odd i, cActive p s key i →
      cInput p s key odd i < bounds (if key then p.k else p.q))
    (output : s.pids 0 < p.Q_HEAD_NUM → ∀ key odd i, cActive p s key i →
      cOffset p s key odd i < bounds (cRegion p key)) :
    TraceSafe (algebra rest ops) bounds (Kernels.cached p) s := by
  by_cases head : s.pids 0 < p.Q_HEAD_NUM
  all_goals
    simp [TraceSafe,Kernels.cached,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
      safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
      algebra,VeriTile.Bench.Repairs.RotaryEmb.Typed.algebra,numericLt,natLt,numericLe,natLe,numericEq,
      bop,numeric,store,Region.cast,ComputeDType.eraseDType,ofFloat,toFloat,
      integralFloorDiv,integralMod,head,token]
    try dsimp only [TileShape.insertAxis,TileShape.dropInsertedIndex]
    all_goals simp only [int_half,←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv,
      Int.toNat_natCast,Int.zero_sub,Int.toNat_neg_natCast,Nat.sub_zero,Int.ofNat_lt]
    all_goals try simp only [←IntegerExpr.tdiv_formula,Int.tmod_def]
  repeat' constructor
  all_goals first
    | exact length head
    | (simpa [cPast,←IntegerExpr.tdiv_formula,ite_mul,neg_sub,sub_lt_iff_lt_add] using table head)
    | (intro d hd; simpa [cCoefficient] using coeff head Bool.false (d,PUnit.unit) hd)
    | (intro d hd; simpa [cCoefficient] using coeff head Bool.true (d,PUnit.unit) hd)
    | (intro d hd; simpa [cInput,cDim] using input head Bool.false Bool.false (d,PUnit.unit) ⟨Or.inl rfl,hd⟩)
    | (intro d hd; simpa [cInput,cDim] using input head Bool.false Bool.true (d,PUnit.unit) ⟨Or.inl rfl,hd⟩)
    | (intro d hk hd; simpa [cInput,cDim] using input head Bool.true Bool.false (d,PUnit.unit) ⟨Or.inr hk,hd⟩)
    | (intro d hk hd; simpa [cInput,cDim] using input head Bool.true Bool.true (d,PUnit.unit) ⟨Or.inr hk,hd⟩)
    | (intro d hk hd; simpa [cOffset,cRegion,cDim,cPast,cBlock,token,←IntegerExpr.tdiv_formula,Int.tmod_def,ite_mul,neg_sub,sub_lt_iff_lt_add] using
        output head Bool.true Bool.false (d,PUnit.unit) ⟨Or.inr hk,hd⟩)
    | (intro d hk hd; simpa [cOffset,cRegion,cDim,cPast,cBlock,token,←IntegerExpr.tdiv_formula,Int.tmod_def,ite_mul,neg_sub,sub_lt_iff_lt_add] using
        output head Bool.true Bool.true (d,PUnit.unit) ⟨Or.inr hk,hd⟩)
    | (intro d hd; simpa [cOffset,cRegion,cDim,cInput] using output head Bool.false Bool.false (d,PUnit.unit) ⟨Or.inl rfl,hd⟩)
    | (intro d hd; simpa [cOffset,cRegion,cDim,cInput] using output head Bool.false Bool.true (d,PUnit.unit) ⟨Or.inl rfl,hd⟩)

#axiomsClean uncached_safe
#axiomsClean cached_safe
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.Typed
