import bench.repairs.tritonbench_g.matmul_kernel.Proofs.PublicIO
import VeriTile.Triton.Float.Loop
import VeriTile.Triton.Float.Equivalence

namespace VeriTile.Bench.Repairs.MatmulKernel.Pair
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem tail_test (lane offset bound : Nat) :
    decide (lane+offset<bound)=decide (lane<bound-offset) := by
  by_cases h : lane+offset<bound
  · have hh : lane<bound-offset := by omega
    simp [h,hh]
  · have hh : ¬lane<bound-offset := by omega
    simp [h,hh]

private theorem eval_lt_nat (M : Algebra α) (p : Option ComputeDType)
    (bc : Broadcast a b c) (x : Op .nat a) (y : Op .nat b) (s : State α) :
    evalOp M p (.lt .nat bc x y) s = (do
      let v ← evalOp M p x s
      let w ← evalOp M p y s
      pure (bop (fun a b : Nat => decide (a<b)) bc v w)) := rfl
private theorem eval_add_nat (M : Algebra α) (p : Option ComputeDType)
    (bc : Broadcast a b c) (x : Op .nat a) (y : Op .nat b) (s : State α) :
    evalOp M p (.add .nat bc x y) s = (do
      let v ← evalOp M p x s
      let w ← evalOp M p y s
      pure (bop Nat.add bc v w)) := rfl
private theorem eval_sub_nat (M : Algebra α) (p : Option ComputeDType)
    (bc : Broadcast a b c) (x : Op .nat a) (y : Op .nat b) (s : State α) :
    evalOp M p (.sub .nat bc x y) s = (do
      let v ← evalOp M p x s
      let w ← evalOp M p y s
      pure (bop Nat.sub bc v w)) := rfl
private theorem eval_const_nat (M : Algebra α) (p : Option ComputeDType)
    (n : Nat) (s : State α) : evalOp M p (.constNat n) s=some (fun _ => n) := rfl

/-- The tail predicate equality is exact discrete arithmetic and preserves
missing-register failures. No floating operation is inspected. -/
theorem tail_mask_eval (M : Algebra α) (precision : Option ComputeDType)
    {dim : Nat} {tail : TileShape} (lane : Op .nat (dim::tail)) (offset : Op .nat [])
    (bound : Nat) (s : State α) :
    FP.Structural.evalOp M precision (.lt .nat .scalarR (.add .nat .scalarR lane offset) (.constNat bound)) s=
      FP.Structural.evalOp M precision (.lt .nat .scalarR lane
        (.sub .nat .nil (.constNat bound) offset)) s := by
  rw [eval_lt_nat,eval_lt_nat,eval_add_nat,eval_sub_nat,eval_const_nat]
  cases ha : evalOp M precision lane s <;>
    cases hb : evalOp M precision offset s <;>
      simp only [ha,hb,Option.bind_some,Option.bind_none,pure,bind]
  apply congrArg some
  funext i
  exact tail_test _ _ _

def candidateBody (p : Config) : List ComputeStmt :=
  match (remaining p).surfaceBody[8]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def firstA (p : Config) := ((body p)[0]?).getD (.ifThen (.constBool Bool.false) [])
def secondB (p : Config) := ((body p)[1]?).getD (.ifThen (.constBool Bool.false) [])
def candidateA (p : Config) := ((candidateBody p)[0]?).getD (.ifThen (.constBool Bool.false) [])
def candidateB (p : Config) := ((candidateBody p)[1]?).getD (.ifThen (.constBool Bool.false) [])
def commonTail (p : Config) := (body p).drop 2

theorem original_body_decomposition (p : Config) :
    body p=firstA p::secondB p::commonTail p := rfl
theorem candidate_body_decomposition (p : Config) :
    candidateBody p=candidateA p::candidateB p::commonTail p := rfl

private theorem eval_load_ptr (M : Algebra α) (p : Option ComputeDType)
    (ptr : Op .ptr sh) (mask : Op .bool sh) (fallback : Op .real sh)
    (padding : PaddingOption) (s : State α) :
    evalOp M p (.load .real (.ptr ptr) (.maskOther mask fallback) padding) s = (do
      let addresses ← evalOp M p ptr s
      let active ← evalOp M p mask s
      let other ← evalOp M p fallback s
      pure (fun i => if active i then (s.mem (addresses i).1 (addresses i).2).read .real else other i)) := rfl
private theorem eval_remap (M : Algebra α) (p : Option ComputeDType)
    (sh : TileShape) (f : TileIndex sh → TileIndex old) (e : Op d old) (s : State α) :
    evalOp M p (.remap sh f e) s = (do
      let v ← evalOp M p e s
      pure (fun i => v (f i))) := rfl

theorem a_step_equal (M : Algebra α) (p : Config) (s : State α) :
    FP.Structural.step M (firstA p) s=FP.Structural.step M (candidateA p) s := by
  simp [firstA,candidateA,KernelSource.body,candidateBody,KernelSource.source,
    KernelSource.typed,remaining,typedRemaining,ComputeKernel.surfaceBody,
    FP.Structural.step,evalExpr,evalComputeOp]
  erw [eval_load_ptr,eval_load_ptr,eval_remap,eval_remap,tail_mask_eval]

theorem b_step_equal (M : Algebra α) (p : Config) (s : State α) :
    FP.Structural.step M (secondB p) s=FP.Structural.step M (candidateB p) s := by
  simp [secondB,candidateB,KernelSource.body,candidateBody,KernelSource.source,
    KernelSource.typed,remaining,typedRemaining,ComputeKernel.surfaceBody,
    FP.Structural.step,evalExpr,evalComputeOp]
  erw [eval_load_ptr,eval_load_ptr,eval_remap,eval_remap,tail_mask_eval]

theorem a_safe_equal (M : Algebra α) (p : Config) (bounds : RegionName → Nat) (s : State α) :
    Safety.stepSafe M bounds (firstA p) s ↔ Safety.stepSafe M bounds (candidateA p) s := by
  simp [firstA,candidateA,KernelSource.body,candidateBody,KernelSource.source,
    KernelSource.typed,remaining,typedRemaining,ComputeKernel.surfaceBody,
    Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.maskActive,eval_remap,tail_mask_eval]

theorem b_safe_equal (M : Algebra α) (p : Config) (bounds : RegionName → Nat) (s : State α) :
    Safety.stepSafe M bounds (secondB p) s ↔ Safety.stepSafe M bounds (candidateB p) s := by
  simp [secondB,candidateB,KernelSource.body,candidateBody,KernelSource.source,
    KernelSource.typed,remaining,typedRemaining,ComputeKernel.surfaceBody,
    Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.maskActive,eval_remap,tail_mask_eval]

theorem body_equal (M : Algebra α) (p : Config) (s : State α) :
    FP.Structural.run M (body p) s=FP.Structural.run M (candidateBody p) s := by
  rw [original_body_decomposition,candidate_body_decomposition]
  rw [FP.Structural.run,FP.Structural.run,a_step_equal]
  apply congrArg (fun f => (FP.Structural.step M (candidateA p) s).bind f)
  funext t
  rw [FP.Structural.run,FP.Structural.run,b_step_equal]

theorem body_safe_equal (M : Algebra α) (p : Config)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (body p) s ↔
      Safety.traceSafeList M bounds (candidateBody p) s := by
  rw [original_body_decomposition,candidate_body_decomposition,
    Safety.traceSafeList_cons,Safety.traceSafeList_cons,a_safe_equal,a_step_equal]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  rw [Safety.traceSafeList_cons,Safety.traceSafeList_cons,b_safe_equal,b_step_equal]

theorem range_equal (M : Algebra α) (p : Config)
    (start stop stride : Nat) (s : State α) :
    FP.Structural.range M "k" start stop stride (body p) s=
      FP.Structural.range M "k" start stop stride (candidateBody p) s := by
  apply range_congr_on_reachable
  intro cur t _ _ _
  exact body_equal M p _

theorem range_safe_equal (M : Algebra α) (p : Config) (bounds : RegionName → Nat)
    (start stop stride : Nat) (s : State α) :
    Safety.rangeSafe M bounds "k" start stop stride (body p) s ↔
      Safety.rangeSafe M bounds "k" start stop stride (candidateBody p) s := by
  induction start using (measure (fun i => stop-i)).wf.induction generalizing s with
  | h cur ih =>
    rw [Safety.rangeSafe_unfold,Safety.rangeSafe_unfold]
    by_cases hz : stride=0
    · simp [hz]
    simp only [hz,if_false]
    by_cases hc : cur<stop
    · simp only [hc,if_true,body_safe_equal,← body_equal]
      cases hr : FP.Structural.run M (body p) (s.setReg "k" .nat [] (fun _ => cur)) with
      | none => rfl
      | some t => exact and_congr Iff.rfl (ih (cur+stride) (by simp_wf;omega) t)
    · simp [hc]

theorem candidate_decomposition (p : Config) :
    (remaining p).surfaceBody=prelude p++[.forRange "k" 0 p.steps 1 (candidateBody p)]++epilogue p := rfl

theorem kernel_equal (M : Algebra α) (p : Config) (s : State α) :
    FP.Structural.exec M (KernelSource.source p) s=FP.Structural.exec M (remaining p) s := by
  change FP.Structural.run M (KernelSource.source p).surfaceBody s=
    FP.Structural.run M (remaining p).surfaceBody s
  rw [source_decomposition,candidate_decomposition]
  simp only [run_append]
  cases hs : FP.Structural.run M (prelude p) s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some]
    have hloop : FP.Structural.run M [.forRange "k" 0 p.steps 1 (body p)] t=
        FP.Structural.run M [.forRange "k" 0 p.steps 1 (candidateBody p)] t := by
      simp only [FP.Structural.run,FP.Structural.step,range_equal]
    rw [hloop]

theorem kernel_safe_equal (M : Algebra α) (p : Config)
    (bounds : RegionName → Nat) (s : State α) :
    TraceSafe M bounds (KernelSource.source p) s ↔ TraceSafe M bounds (remaining p) s := by
  unfold TraceSafe
  rw [source_decomposition,candidate_decomposition,List.append_assoc,List.append_assoc,
    Safety.traceSafeList_append,Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  rw [List.singleton_append,List.singleton_append,Safety.traceSafeList_cons,Safety.traceSafeList_cons]
  simp only [Safety.stepSafe_unfold,FP.Structural.step,range_safe_equal,← range_equal]

theorem invocation_equal (p : Config) (M : Algebra α) (D : DotAcc α) (pid : Program) (s : State α) :
    (PublicIO.invocation p M D pid).run s=(invocation p M D pid).run s :=
  kernel_equal (backend M D) p _

theorem invocation_safe_equal (p : Config) (M : Algebra α) (D : DotAcc α) (pid : Program)
    (bounds : RegionName → Nat) (s : State α) :
    (PublicIO.invocation p M D pid).Safe bounds s ↔ (invocation p M D pid).Safe bounds s :=
  kernel_safe_equal (backend M D) p bounds _

theorem schedule_equal (p : Config) (M : Algebra α) (D : DotAcc α) (ids : List Program) (s : State α) :
    runSchedule (ids.map (PublicIO.invocation p M D)) s=runSchedule (ids.map (invocation p M D)) s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,invocation_equal]
    cases hr : (invocation p M D pid).run s with
    | none => rfl
    | some t => exact ih t

theorem schedule_safe_equal (p : Config) (M : Algebra α) (D : DotAcc α) (ids : List Program)
    (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (ids.map (PublicIO.invocation p M D)) s ↔
      ScheduleSafe bounds (ids.map (invocation p M D)) s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_equal]
    exact and_congr (invocation_safe_equal p M D pid bounds s)
      (forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

theorem implements_transfer (p : Config) (M : Algebra α) (D : DotAcc α)
    (f : (PublicIO.io p M D).Inputs → (PublicIO.io p M D).Outputs)
    (original : (PublicIO.io p M D).Implements f) : (candidateIO p M D).Implements f := by
  intro vals hv bounds hb initial hin
  obtain ⟨safe,final,ran,outputs,frame⟩ := original vals hv bounds hb initial hin
  exact ⟨(schedule_safe_equal p M D _ bounds initial).mp safe,final,
    (schedule_equal p M D _ initial).symm.trans ran,outputs,frame⟩

#axiomsClean kernel_equal
#axiomsClean kernel_safe_equal
#axiomsClean implements_transfer

end
end VeriTile.Bench.Repairs.MatmulKernel.Pair
