import bench.tritonbench_g.fused_rotary_embedding.Proofs.RawProgram

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Bounds for every active access, including the metadata load that determines
the indirect cache addresses. These are proved from the public IO windows. -/
structure AccessBounds (p : Parameters) (s : State α) (bounds : RegionName → Nat) : Prop where
  q : ∀ high i, active p i → dataOffset p s Bool.false high i < bounds "q"
  coeff : ∀ sine i, active p i →
    coefficientOffset p s sine i < bounds (if sine then "sin" else "cos")
  k : s.pids 0%p.KV_GROUP_NUM=0 → ∀ high i, active p i →
    dataOffset p s Bool.true high i < bounds "k"
  length : s.pids 0%p.KV_GROUP_NUM=0 →
    ((s.pids 1 : Int)*p.length_stride).toNat < bounds "context_lengths"
  table : s.pids 0%p.KV_GROUP_NUM=0 →
    ((s.pids 1 : Int)*p.bts_stride+quot (past p s) p.block_size*p.btb_stride).toNat < bounds "BLOCK_TABLES"
  kcache : s.pids 0%p.KV_GROUP_NUM=0 → ∀ high i, active p i → cacheOffset p s high i < bounds "k_cache"
  v : s.pids 0%p.KV_GROUP_NUM=0 → ∀ i : FullLane p, i.1.val<p.HEAD_DIM → valueOffset p s i < bounds "v"
  vcache : s.pids 0%p.KV_GROUP_NUM=0 → ∀ i : FullLane p, i.1.val<p.HEAD_DIM → valueCacheOffset p s i < bounds "v_cache"

theorem source_safe (rest : Algebra α) (ops : Ops α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) (h : AccessBounds p s bounds) :
    TraceSafe (algebra rest ops) bounds (kernel p) s := by
  have headNonneg : ¬(s.pids 0 : Int)<0 := by omega
  have signedLeader :
      ((s.pids 0 : Int)-(p.KV_GROUP_NUM : Int)*((s.pids 0 : Int)/(p.KV_GROUP_NUM : Int))=0) ↔
        s.pids 0%p.KV_GROUP_NUM=0 := by
    rw [←Int.emod_def,←Int.natCast_emod,Int.natCast_eq_zero]
  have hq0 := fun i : Fin p.BLOCK_HALF => h.q Bool.false (i,PUnit.unit)
  have hq1 := fun i : Fin p.BLOCK_HALF => h.q Bool.true (i,PUnit.unit)
  have hc := fun i : Fin p.BLOCK_HALF => h.coeff Bool.false (i,PUnit.unit)
  have hs := fun i : Fin p.BLOCK_HALF => h.coeff Bool.true (i,PUnit.unit)
  simp only [active,dataOffset,coefficientOffset,cacheOffset,valueOffset,valueCacheOffset,past,block,slot,head,dimension,quot,rem,headNonneg,Bool.false_eq_true,ite_true,ite_false,Int.add_zero] at hq0 hq1 hc hs
  by_cases leader : s.pids 0%p.KV_GROUP_NUM=0
  · have hk0 := fun i : Fin p.BLOCK_HALF => h.k leader Bool.false (i,PUnit.unit)
    have hk1 := fun i : Fin p.BLOCK_HALF => h.k leader Bool.true (i,PUnit.unit)
    have hl := h.length leader
    have ht := h.table leader
    have hkc0 := fun i : Fin p.BLOCK_HALF => h.kcache leader Bool.false (i,PUnit.unit)
    have hkc1 := fun i : Fin p.BLOCK_HALF => h.kcache leader Bool.true (i,PUnit.unit)
    have hv := fun i : Fin p.BLOCK_DIM => h.v leader (i,PUnit.unit)
    have hvc := fun i : Fin p.BLOCK_DIM => h.vcache leader (i,PUnit.unit)
    simp only [active,dataOffset,coefficientOffset,cacheOffset,valueOffset,valueCacheOffset,past,block,slot,head,dimension,quot,rem,headNonneg,Bool.false_eq_true,ite_true,ite_false,Int.add_zero] at hk0 hk1 hl ht hkc0 hkc1 hv hvc
    simp at ht hkc0 hkc1 hv hvc
    simp (maxSteps := 1000000) [TraceSafe,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,Safety.safeAlg_unfold,
      kernel,ComputeKernel.surfaceBody,FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,evalDeltas,Option.bind,algebra,numericLt,natLt,numericEq,bop,numeric,
      store,VeriTile.Triton.Region.cast,ComputeDType.eraseDType,ofFloat,toFloat,
      integralFloorDiv,integralMod,headNonneg,signedLeader,leader,hq0,hq1,hc,hs,hk0,hk1,hl,ht,hkc0,hkc1,hv,hvc]
    repeat' apply And.intro
    all_goals first | exact hq0 | exact hq1 | exact hc | exact hs | exact hk0 | exact hk1 | exact ht | exact hkc0 | exact hkc1 | exact hv | exact hvc
  · simp (maxSteps := 1000000) [TraceSafe,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,Safety.safeAlg_unfold,
      kernel,ComputeKernel.surfaceBody,FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,evalDeltas,Option.bind,algebra,numericLt,natLt,numericEq,bop,numeric,
      store,VeriTile.Triton.Region.cast,ComputeDType.eraseDType,ofFloat,toFloat,
      integralFloorDiv,integralMod,headNonneg,signedLeader,leader,hq0,hq1,hc,hs]
    exact ⟨hq0,hq1,hc,hs,hq0,hq1⟩

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
