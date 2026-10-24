import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.Launch
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.Fp4ToBf16.Access
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 100000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

private theorem pure_list_safe (rest : Algebra Word) (bounds : RegionName → Nat)
    (code : List ComputeStmt)
    (pure : ∀ st∈code,∀ s : State Word,Safety.stepSafe (backend rest) bounds st s)
    (s : State Word) : Safety.traceSafeList (backend rest) bounds code s := by
  induction code generalizing s with
  | nil => simp [Safety.traceSafeList_unfold]
  | cons st code ih =>
    apply Safety.traceSafeList_cons_intro
    · exact pure st (by simp) s
    · intro t _
      exact ih (fun st member => pure st (by simp [member])) t

private theorem middle_safe (rest : Algebra Word) (p : Parameters) (s : State Word)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList (backend rest) bounds ((source p).surfaceBody.drop 6 |>.take 19) s := by
  apply pure_list_safe
  intro st member t
  simp only [source,unscaled,ComputeKernel.surfaceBody,List.drop,List.take,
    List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Op.intToSigned,Op.intToUnsigned,Op.unsignedToInt]

private theorem load_safe (rest : Algebra Word) (p : Parameters) (s : State Word)
    (bounds : RegionName → Nat)
    (input : ∀ i : Fin p.block,inputOffset p s i.val<p.count →
      inputOffset p s i.val<bounds p.input) :
    Safety.traceSafeList (backend rest) bounds ((source p).surfaceBody.take 6) s := by
  simp [source,unscaled,ComputeKernel.surfaceBody,List.take,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,backend,numeric,FP.Structural.bop,natLt,
    numericLt,Region.cast,Option.bind,ComputeDType.eraseDType]
  exact input

private theorem prefix_run (rest : Algebra Word) (p : Parameters) (s : State Word) :
    run (backend rest) ((source p).surfaceBody.take 25) s=some (signExecution rest p s).val := by
  change run _ (((source p).surfaceBody.take 11)++
    ((source p).surfaceBody.drop 11 |>.take 3)++
    ((source p).surfaceBody.drop 14 |>.take 1)++
    ((source p).surfaceBody.drop 15 |>.take 1)++
    ((source p).surfaceBody.drop 16 |>.take 1)++
    ((source p).surfaceBody.drop 17 |>.take 5)++
    ((source p).surfaceBody.drop 22 |>.take 3)) s=_
  have load : run (backend rest) ((source p).surfaceBody.take 11) s=
      some (loadExecution rest p s).val := by
    simpa only [List.drop_zero] using (loadExecution rest p s).property
  simp only [run_append,load,(maskExecution rest p s).property,
    (minusExecution rest p s).property,(addBiasExecution rest p s).property,
    (expExecution rest p s).property,(normalExecution rest p s).property,
    (signExecution rest p s).property,Option.bind_some]

private theorem tail_safe (rest : Algebra Word) (p : Parameters) (s : State Word)
    (bounds : RegionName → Nat) (pid : Nat) (words : TileIndex [p.block*2] → Int)
    (hw : s.regs .int [p.block*2] "result"=some words)
    (hp : s.regs .nat [] "pid"=some (fun _ => pid))
    (hc : s.regs .nat [] "n_elements_out"=some (fun _ => p.count*2))
    (output : ∀ j : Fin (p.block*2),pid*(p.block*2)+j.val<p.count*2 →
      pid*(p.block*2)+j.val<bounds p.output) :
    Safety.traceSafeList (backend rest) bounds ((source p).surfaceBody.drop 25) s := by
  simp [source,unscaled,ComputeKernel.surfaceBody,List.drop,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,store,backend,hw,hp,hc,
    numeric,FP.Structural.bop,natLt,numericLt,ofFloat,toFloat,Region.cast,Option.bind,
    ComputeDType.eraseDType]
  exact output

/-- Actual-source load and store safety; pure packed-integer conversion is
checked structurally without re-expanding its established execution proof. -/
theorem source_safe (rest : Algebra Word) (p : Parameters) (s : State Word)
    (bounds : RegionName → Nat)
    (input : ∀ i : Fin p.block,inputOffset p s i.val<p.count →
      inputOffset p s i.val<bounds p.input)
    (output : ∀ j : TileIndex [p.block*2],active p s j →
      outputOffset p s j<bounds p.output) :
    TraceSafe (backend rest) bounds (Kernels.source p) s := by
  change Safety.traceSafeList (backend rest) bounds (source p).surfaceBody s
  rw [←List.take_append_drop 25 (source p).surfaceBody,Safety.traceSafeList_append]
  constructor
  · change Safety.traceSafeList (backend rest) bounds
      (((source p).surfaceBody.take 6)++((source p).surfaceBody.drop 6 |>.take 19)) s
    rw [Safety.traceSafeList_append]
    exact ⟨load_safe rest p s bounds input,fun t _ => middle_safe rest p t bounds⟩
  · intro t executed
    have same := Option.some.inj ((prefix_run rest p s).symm.trans executed)
    subst t
    exact tail_safe rest p _ bounds (s.pids 0) (fun j => integerWord (laneNibble p s j))
      (signExecution_result rest p s) (signExecution_pid rest p s)
      (signExecution_n_elements_out rest p s) (fun j live => output (j,PUnit.unit) live)

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.Fp4ToBf16.Access
