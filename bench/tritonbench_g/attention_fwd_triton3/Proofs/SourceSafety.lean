import bench.tritonbench_g.attention_fwd_triton3.Proofs.SourceSafetyGeometry
import bench.tritonbench_g.attention_fwd_triton3.Proofs.SourceSafetyTail

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
open VeriTile Triton FP.Structural Safety FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open Reference
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 32000
variable {α : Type} [Inhabited α]

theorem range_safe (c : Call) (o : Ops α) (positive : 0<c.BN) (bounds : RegionName → Nat)
    (initialState s : State α) (old : Carry c α)
    (live : Matches c o (inputs initialState) old s) (memory : s.mem=initialState.mem)
    (windows : Windows c (inputs initialState) bounds)
    (cur : Nat) (aligned : c.BN∣cur) (pointers : PointerAt c (inputs initialState) old cur) :
    rangeIntSafe (engine o.algebra) bounds "start_n" cur (upper c (inputs initialState)) c.BN
      (parts c).body s := by
  rw [rangeIntSafe_unfold,if_neg (Nat.ne_of_gt positive)]
  by_cases active : (cur : Int)<upper c (inputs initialState)
  · rw [if_pos active]
    have key := key_bounds c (inputs initialState) bounds windows cur aligned active
    have value := value_bounds c (inputs initialState) bounds windows cur aligned active
    rw [←pointers.1] at key
    rw [←pointers.2] at value
    refine ⟨body_safe c o positive bounds initialState s old live cur key value,?_⟩
    obtain ⟨mid,ran,live',memory'⟩ := body c o positive initialState s old live memory cur
    rw [ran]
    simpa only [Nat.cast_add] using range_safe c o positive bounds initialState mid
      (advance c o (inputs initialState) cur old) live' memory' windows (cur+c.BN)
      (dvd_add aligned (dvd_refl _)) (advance_pointer c o (inputs initialState) old cur pointers)
  · rw [if_neg active]
    trivial
termination_by (upper c (inputs initialState)-(cur : Int)).toNat
decreasing_by omega

theorem loop_safe (c : Call) (o : Ops α) (positive : 0<c.BN) (bounds : RegionName → Nat)
    (initialState s : State α)
    (live : Matches c o (inputs initialState) (initial c o (inputs initialState)) s)
    (memory : s.mem=initialState.mem) (windows : Windows c (inputs initialState) bounds) :
    stepSafe (engine o.algebra) bounds
      (.forRangeInt "start_n" (parts c).start (parts c).stop (parts c).stride (parts c).body) s := by
  have safe := range_safe c o positive bounds initialState s _ live memory windows
    (lower c (inputs initialState)).toNat (lower_aligned c _ positive) (initial_pointer c o _)
  rw [Int.toNat_of_nonneg (lower_nonnegative c _)] at safe
  rcases live with ⟨_,_,_,_,_,_,_,_,_,_,_,hlo,hhi⟩
  have lo : evalOp (engine o.algebra) none (parts c).start s=
      some (fun _ => lower c (inputs initialState)) := hlo
  have hi : evalOp (engine o.algebra) none (parts c).stop s=
      some (fun _ => upper c (inputs initialState)) := hhi
  have stride : evalOp (engine o.algebra) none (parts c).stride s=some (fun _ => c.BN) := rfl
  rw [stepSafe_unfold]
  refine ⟨?_,?_,?_,?_⟩
  · change safeOp (engine o.algebra) none bounds s (.ref .int [] "lo")
    rw [safeOp_unfold]
    trivial
  · change safeOp (engine o.algebra) none bounds s (.ref .int [] "hi")
    rw [safeOp_unfold]
    trivial
  · change safeOp (engine o.algebra) none bounds s (.constNat c.BN)
    rw [safeOp_unfold]
    trivial
  · rw [lo,hi,stride]
    exact safe

/-- Actual load/store safety for the whole selected quotation. Branches and
the signed loop follow the existing execution invariants, not a separate model. -/
theorem source_safe (c : Call) (o : Ops α) (positive : 0<c.BN) (bounds : RegionName → Nat)
    (s : State α) (windows : Windows c (inputs s) bounds) :
    TraceSafe (engine o.algebra) bounds (Kernels.source c) s := by
  obtain ⟨pre,ranPre,live,memory⟩ := prelude c o s
  obtain ⟨post,ranLoop,live',memory'⟩ := Reference.loop c o positive s pre live memory
  unfold TraceSafe
  rw [source_split,List.append_assoc,List.singleton_append]
  exact append_of_run (prelude_safe c o bounds s windows.toPreludeBounds) ranPre
    (traceSafeList_cons_of_step (loop_safe c o positive bounds s pre live memory windows) ranLoop
      (epilogue_safe c o bounds (inputs s) (result c o (inputs s)) post live' windows.toPreludeBounds))

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
