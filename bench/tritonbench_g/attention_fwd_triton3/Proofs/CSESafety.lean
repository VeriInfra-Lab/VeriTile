import bench.tritonbench_g.attention_fwd_triton3.Proofs.CSEProgram
import bench.tritonbench_g.attention_fwd_triton3.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE
open VeriTile Triton FP.Structural Safety
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.Repairs.ContextAttnMistral
open Reference SourceSafety
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem body_safe (c : Call) (o : Ops α) (positive : 0<c.BN) (bounds : RegionName → Nat)
    (initialState s : State α) (old : Carry c α) (live : Matches c o (inputs initialState) old s)
    (cur : Int) (memory : s.mem=initialState.mem)
    (key : BlockBounds bounds old.keyPointer [c.D,c.BN] (if c.params.evenN≠0 then [] else [0,1]))
    (value : BlockBounds bounds old.valuePointer [c.BN,c.D] (if c.params.evenN≠0 then [] else [0,1])) :
    traceSafeList (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) bounds (parts c).body
      (s.setReg "__veritile_statement_jit_local_11_start_n" .int [] (fun _ => cur)) := by
  obtain ⟨scored,hscore,scoreReady,scoreMemory⟩:=score_phase c o positive initialState s old live memory cur
  obtain ⟨updated,hupdate,updateReady,updateMemory⟩:=
    update_phase c o positive initialState scored old cur scoreReady scoreMemory
  have first : (parts c).body.take 2=(Reference.parts c).body.take 2 := rfl
  have scoreSafe : traceSafeList (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) bounds
      ((parts c).body.take 7)
      (s.setReg "__veritile_statement_jit_local_11_start_n" .int [] (fun _=>cur)) := by
    have split : (parts c).body.take 7=(parts c).body.take 2 ++ ((parts c).body.drop 2).take 5 := rfl
    rw [split,traceSafeList_append]
    refine ⟨?_,?_⟩
    · rw [first]
      exact SourceSafety.score_load_safe c o positive bounds initialState s old live cur key
    · intro t _
      prove_memory_free
  have updateSafe : traceSafeList (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) bounds
      (((parts c).body.drop 7).take 12) scored := by prove_memory_free
  have productSafe : traceSafeList (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) bounds
      ((parts c).body.drop 19) updated := by
    have same : (parts c).body.drop 19=(Reference.parts c).body.drop 19 := rfl
    rw [same]
    exact SourceSafety.product_safe c o positive bounds initialState updated
      (scaledCarry c o (inputs initialState) cur old) updateReady.1 value
  have split : (parts c).body=(parts c).body.take 7 ++
      (((parts c).body.drop 7).take 12 ++ (parts c).body.drop 19) := rfl
  rw [split]
  exact append_of_run scoreSafe hscore (append_of_run updateSafe hupdate productSafe)

theorem range_safe (c : Call) (o : Ops α) (positive : 0<c.BN) (bounds : RegionName → Nat)
    (initialState s : State α) (old : Carry c α)
    (live : Matches c o (inputs initialState) old s) (memory : s.mem=initialState.mem)
    (windows : Windows c (inputs initialState) bounds)
    (cur : Nat) (aligned : c.BN∣cur) (pointers : PointerAt c (inputs initialState) old cur) :
    rangeIntSafe (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) bounds "__veritile_statement_jit_local_11_start_n" cur (upper c (inputs initialState)) c.BN
      (parts c).body s := by
  rw [rangeIntSafe_unfold,if_neg (Nat.ne_of_gt positive)]
  by_cases active : (cur : Int)<upper c (inputs initialState)
  · rw [if_pos active]
    have key := key_bounds c (inputs initialState) bounds windows cur aligned active pointers.2.2
    have value := value_bounds c (inputs initialState) bounds windows cur aligned active pointers.2.2
    rw [←pointers.1] at key
    rw [←pointers.2.1] at value
    refine ⟨body_safe c o positive bounds initialState s old live cur memory key value,?_⟩
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
    stepSafe (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) bounds
      (.forRangeInt "__veritile_statement_jit_local_11_start_n" (parts c).start (parts c).stop (parts c).stride (parts c).body) s := by
  have safe := range_safe c o positive bounds initialState s _ live memory windows
    (lower c (inputs initialState)).toNat (lower_aligned c _ positive) (initial_pointer c o _)
  rw [Int.toNat_of_nonneg (lower_nonnegative c _)] at safe
  rcases live with ⟨_,_,_,_,_,_,_,_,_,_,_,hlo,hhi,_,_,_,_⟩
  have lo : evalOp (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) none (parts c).start s=
      some (fun _ => lower c (inputs initialState)) := hlo
  have hi : evalOp (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) none (parts c).stop s=
      some (fun _ => upper c (inputs initialState)) := hhi
  have stride : evalOp (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) none (parts c).stride s=some (fun _ => c.BN) := rfl
  rw [stepSafe_unfold]
  refine ⟨?_,?_,?_,?_⟩
  · change safeOp (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) none bounds s (.ref .int [] "__veritile_statement_jit_local_11_lo")
    rw [safeOp_unfold]
    trivial
  · change safeOp (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) none bounds s (.ref .int [] "__veritile_statement_jit_local_11_hi")
    rw [safeOp_unfold]
    trivial
  · change safeOp (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) none bounds s (.constNat c.BN)
    rw [safeOp_unfold]
    trivial
  · rw [lo,hi,stride]
    exact safe

/-- Actual load/store safety for the whole selected quotation. Branches and
the signed loop follow the existing execution invariants, not a separate model. -/
theorem source_safe (c : Call) (o : Ops α) (positive : 0<c.BN) (bounds : RegionName → Nat)
    (s : State α) (windows : Windows c (inputs s) bounds) :
    TraceSafe (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) bounds (Kernels.keyMaskCSE c) s := by
  obtain ⟨pre,ranPre,live,memory⟩ := prelude c o s
  obtain ⟨post,ranLoop,live',memory'⟩ := KeyMaskCSE.loop c o positive s pre live memory
  unfold TraceSafe
  rw [source_split,List.append_assoc,List.singleton_append,before_eq,after_eq]
  exact append_of_run (prelude_safe c o bounds s windows.toPreludeBounds) ranPre
    (traceSafeList_cons_of_step (loop_safe c o positive bounds s pre live memory windows) ranLoop
      (epilogue_safe c o bounds (inputs s) (result c o (inputs s)) post live' windows.toPreludeBounds))

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE
