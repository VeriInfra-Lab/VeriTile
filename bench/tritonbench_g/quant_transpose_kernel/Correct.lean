import bench.tritonbench_g.quant_transpose_kernel.Kernels
import bench.tritonbench_g.quant_transpose_kernel.Proofs.RealHost

/-! Independent Real specification and proof for the original transpose
wrapper, including source-ordered setup failures, grouped trials, returned
tensor metadata, mathematical undefined observations and memory frame. -/
namespace QuantTransposeReal
open QuantTransposeFinite (Input Allocations Config Parameters parameters Stage Environment)
open QuantizeGlobalReal (Cell Memory Write Issue Tensor PayloadKind Prepared maximumOf
  readInputs Outside outside_refl writes_outside finalCell writes_final_cell writes_frame)
noncomputable section
open Classical
set_option maxHeartbeats 12000000
set_option maxRecDepth 64000

specification quantize_global_transpose_real_source_outcomes
    (env : Environment) (a : Input) (alloc : Allocations) (m : Memory) (x : String)
    (metadata : Metadata a) (fresh : alloc.Fresh x) (trials : List Config) (last : Config) :
    Metadata a ∧ HostSpec env a alloc m x trials last
      (executeHost env a alloc m x trials last) := by
  refine ⟨metadata,?_⟩
  have prep := prepare_all_outcomes env a alloc m x
  cases hp : prepare env a alloc m x with
  | error why => simpa [executeHost,hp,HostSpec,PreparationSpec] using prep
  | ok p =>
    have pp : PreparedSpec env a alloc m x p := by simpa [hp,PreparationSpec] using prep
    have set := setup_all_outcomes env a
    cases hs : setup env a with
    | error why =>
      have failure : SetupFailure env a why := by simpa [hs] using set
      simpa [executeHost,hp,hs,HostSpec] using (show ∃ q, PreparedSpec env a alloc m x q ∧ SetupFailure env a why from ⟨p,pp,failure⟩)
    | ok done =>
      cases done
      have ready : SetupReady env a := by simpa [hs] using set
      cases exit : env .launchExitHook with
      | some detail =>
        simp only [executeHost,hp,hs,exit,HostSpec]
        exact ⟨p,pp,ready,True.intro⟩
      | none =>
        have corr := trials_correspondence a alloc x p.initial (trials++[last]) fresh p.initial (outside_refl _ _)
        cases ht : trialsPlan a alloc x p.initial (trials++[last]) with
        | error issue =>
          simp only [ht,Except.map] at corr
          simp only [executeHost,hp,hs,exit,corr,HostSpec]
          exact ⟨p,pp,ready,True.intro,ht⟩
        | ok writes =>
          simp only [ht,Except.map] at corr
          simp only [executeHost,hp,hs,exit,corr,HostSpec]
          have allocation : p.initial=allocated a alloc m p.maximum p.inverse := by
            rcases pp with ⟨_,_,_,_,_,_,_,values,_,_,_,h⟩
            exact h
          have count : a.numel=(parameters a alloc x).n := by
            exact matrix_count a pp.2.2.2.2.2.1
          refine ⟨rfl,p,pp,ready,exit,writes,ht,?_,?_,?_,?_,?_,?_,?_,?_⟩
          · exact fun r i => writes_final_cell _ _ _ _ _
          · exact fun r i h => writes_frame _ _ _ _ _ h
          · intro w hw
            have wi : w.1∈writes.map Prod.fst := List.mem_map.mpr ⟨w,hw,rfl⟩
            rw [trials_plan_indices _ _ _ _ _ _ ht] at wi
            rw [count]
            exact trial_indices_bound _ _ _ wi
          · intro i hi
            rw [count] at hi
            rw [trials_plan_indices _ _ _ _ _ _ ht]
            exact complete_trials_index _ _ _ _ hi
          · exact writes_outside _ _ _
          · rw [writes_outside _ _ _ _ _ fresh.2.2.2.2.1,allocation]
            simp [allocated,QuantizeGlobalReal.allocated,allocations]
          · rw [writes_outside _ _ _ _ _ fresh.2.2.2.2.2,allocation]
            simp [allocated,QuantizeGlobalReal.allocated,allocations,Ne.symm fresh.2.2.2.1]
          · intro r i hm hs ho
            rw [writes_outside _ _ _ _ _ ho,allocation]
            simp [allocated,QuantizeGlobalReal.allocated,allocations,hm,hs,ho]

#axiomsClean quantize_global_transpose_real_source_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantTransposeReal
