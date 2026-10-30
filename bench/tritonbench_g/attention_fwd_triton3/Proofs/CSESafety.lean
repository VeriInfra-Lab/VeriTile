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
    (cur : Int)
    (key : BlockBounds bounds old.keyPointer [c.D,c.BN] (if c.params.evenN≠0 then [] else [0,1]))
    (value : BlockBounds bounds old.valuePointer [c.BN,c.D] (if c.params.evenN≠0 then [] else [0,1])) :
    traceSafeList (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) bounds (parts c).body
      (s.setReg "start_n" .int [] (fun _ => cur)) := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12⟩
  by_cases he : c.params.evenN=0 <;> by_cases hw : c.window=0 <;> by_cases hc : c.complement=0
  all_goals
    simp (config := {maxSteps := 1000000}) [parts,Kernels.keyMaskCSE,Quoted.keyMaskCSE,
      ComputeKernel.surfaceBody,Source.intRange?,traceSafeList_unfold,stepSafe_unfold,
      safeExpr_unfold,safeCompute_unfold,safeOp_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      State.setReg,VeriTile.Triton.FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,FPBackend.Ops.algebra,resolvePrecision,
      numeric,numericLt,numericLe,numericEq,natLt,natLe,ofFloat,toFloat,Region.cast,boundaryZero,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,he,hw,hc,positive,TileShape.axisDim,
      bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    simp only [block_bounds_2d,he,ne_eq,not_true_eq_false,not_false_eq_true,if_true,if_false,BlockPtr.inBounds_nil_checkedAxes,forall_true_left] at key value
  all_goals exact ⟨key,value⟩


theorem range_safe (c : Call) (o : Ops α) (positive : 0<c.BN) (bounds : RegionName → Nat)
    (initialState s : State α) (old : Carry c α)
    (live : Matches c o (inputs initialState) old s) (memory : s.mem=initialState.mem)
    (windows : Windows c (inputs initialState) bounds)
    (cur : Nat) (aligned : c.BN∣cur) (pointers : PointerAt c (inputs initialState) old cur) :
    rangeIntSafe (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) bounds "start_n" cur (upper c (inputs initialState)) c.BN
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
    stepSafe (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) bounds
      (.forRangeInt "start_n" (parts c).start (parts c).stop (parts c).stride (parts c).body) s := by
  have safe := range_safe c o positive bounds initialState s _ live memory windows
    (lower c (inputs initialState)).toNat (lower_aligned c _ positive) (initial_pointer c o _)
  rw [Int.toNat_of_nonneg (lower_nonnegative c _)] at safe
  rcases live with ⟨_,_,_,_,_,_,_,_,_,_,_,hlo,hhi⟩
  have lo : evalOp (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) none (parts c).start s=
      some (fun _ => lower c (inputs initialState)) := hlo
  have hi : evalOp (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) none (parts c).stop s=
      some (fun _ => upper c (inputs initialState)) := hhi
  have stride : evalOp (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) none (parts c).stride s=some (fun _ => c.BN) := rfl
  rw [stepSafe_unfold]
  refine ⟨?_,?_,?_,?_⟩
  · change safeOp (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) none bounds s (.ref .int [] "lo")
    rw [safeOp_unfold]
    trivial
  · change safeOp (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) none bounds s (.ref .int [] "hi")
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
