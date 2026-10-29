import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.NativeSafetyLoop
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIOValues

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeInterface.SafetyProof
open VeriTile Triton FP.Structural Reused Geometry NativeStaging
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000

theorem capacities (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (xs : Inputs c) (bounds : RegionName → Nat)
    (hb : (programIO c offset size scale window complement).InBounds bounds xs) :
    ∀ r a, a < DenseSafety.bounds c r → a < bounds r := by
  intro r a ha
  by_cases hq : r="q"
  · subst r
    exact hb.2.1 (0 : Fin 4) (⟨a,by simpa [DenseSafety.bounds] using ha⟩,()) (Or.inl (by change (0 : Nat)≠3; decide))
  by_cases hk : r="k"
  · subst r
    exact hb.2.1 (1 : Fin 4) (⟨a,by simpa [DenseSafety.bounds] using ha⟩,()) (Or.inl (by change (1 : Nat)≠3; decide))
  by_cases hm : r="m"
  · subst r
    exact hb.2.1 (2 : Fin 4) (⟨a,by simpa [DenseSafety.bounds] using ha⟩,()) (Or.inl (by change (2 : Nat)≠3; decide))
  by_cases ho : r="out"
  · subst r
    exact hb.2.2.1 (0 : Fin 1) (⟨a,by simpa [DenseSafety.bounds] using ha⟩,()) trivial
  simp [DenseSafety.bounds,hq,hk,hm,ho] at ha

theorem source_bounds (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (s : State NativeStaging.Expr) (job : WholeGrid.Job c) (bounds : RegionName → Nat)
    (cap : ∀ r a, a < DenseSafety.bounds c r → a < bounds r) :
    NativeSafety.Bounds c.B c.D (parameters c offset size scale window complement)
      (FPGrid.programState c s job) bounds := by
  obtain ⟨bq,bk,bm⟩ := Binding.bases c offset size scale window complement s job
  constructor
  · intro d j active
    have live : job.2.2.val*c.B+j.val<c.NK := by
      rcases active with even|live
      · exact DenseSafety.full_row _ _ _ _ (supported_positive c.B c.block) job.2.2.isLt j.isLt
          (of_decide_eq_true even)
      · exact live
    change FPContext.baseK (parameters c offset size scale window complement) (FPGrid.programState c s job)+
      d.val*1+(job.2.2.val*c.B+j.val)*c.D < bounds "k"
    rw [bk]
    apply cap
    simpa [DenseSafety.bounds,Nat.mul_one,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
      key_bound c job.1 job.2.1 ⟨job.2.2.val*c.B+j.val,live⟩ d
  · intro row aligned live i d active
    have hr : row+i.val<c.N := by
      rcases active with even|hr
      · exact Binding.full_row _ _ _ (supported_positive c.B c.block) aligned live
          (of_decide_eq_true even) i
      · exact hr
    change FPContext.baseQ (parameters c offset size scale window complement) (FPGrid.programState c s job)+
      (row+i.val)*c.D+d.val*1 < bounds "q"
    rw [bq]
    apply cap
    simpa [DenseSafety.bounds,Nat.mul_one] using query_bound c job.1 job.2.1 ⟨row+i.val,hr⟩ d
  · intro row live
    change FPContext.baseM (parameters c offset size scale window complement) (FPGrid.programState c s job)+
      row < bounds "m"
    rw [bm]
    exact cap _ _ (normalizer_bound c job.1 job.2.1 ⟨row,live⟩)
  · intro i live
    rw [FPGrid.output_address]
    apply cap
    simpa [WholeGrid.address,DenseSafety.bounds,Nat.add_assoc] using
      output_bound c job.1 job.2.1 ⟨job.2.2.val*c.B+i.val,live⟩

theorem invocation_safe (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (job : WholeGrid.Job c) (s : State NativeStaging.Expr) (bounds : RegionName → Nat)
    (cap : ∀ r a, a < DenseSafety.bounds c r → a < bounds r)
    (hOffsets : KeyOffsetsFit c) :
    (invocation c offset size scale window complement job).Safe bounds s := by
  unfold Invocation.Safe
  rw [invocation_initial]
  change TraceSafe _ bounds (kernel c offset size scale window complement) _
  rw [shared_source]
  exact NativeSafety.source_safe algebra dotModel c.B c.D (parameters c offset size scale window complement)
    (FPGrid.programState c s job) bounds (source_bounds c offset size scale window complement s job bounds cap)
    (supported_positive c.B c.block) (by simpa [FPGrid.programState] using hOffsets job.2.2)

theorem schedule_safe (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (order : List (WholeGrid.Job c)) (s : State NativeStaging.Expr) (bounds : RegionName → Nat)
    (cap : ∀ r a, a < DenseSafety.bounds c r → a < bounds r)
    (hOffsets : KeyOffsetsFit c) :
    ScheduleSafe bounds (order.map (invocation c offset size scale window complement)) s := by
  induction order generalizing s with
  | nil => trivial
  | cons job rest ih =>
    exact ⟨invocation_safe c offset size scale window complement job s bounds cap hOffsets,fun next _ => ih next⟩

#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeInterface.SafetyProof
