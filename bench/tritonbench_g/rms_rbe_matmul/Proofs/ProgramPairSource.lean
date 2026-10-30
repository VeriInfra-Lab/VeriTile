import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelPrefix
import bench.tritonbench_g.cache_transform_triton.Sources.Mathematical_IndexRewrite
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.SourcePair
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Complete source identity, including ports, all typed instructions and branches. -/
theorem readable_preserves (fp8 : Bool) (p : TypedSource.Arguments) :
    TypedSource.source fp8 p = RmsMatmulRbe.TypedSource.kernel fp8 fp8 p := by
  cases fp8 <;> rfl

def original (fp8 : Bool) (p : TypedSource.Arguments) : FP.ProfiledRewrite.Program :=
  ⟨TypedSource.source fp8 p, []⟩
def indexCommuted (fp8 : Bool) (p : TypedSource.Arguments) : FP.ProfiledRewrite.Program :=
  ⟨TypedSource.indexCommutedKernel fp8 p, []⟩

/-- The historical exact address rewrite now covers the actual repaired source. -/
theorem equivalent (fp8 : Bool) (p : TypedSource.Arguments) :
    original fp8 p ≡[([] : Spec.Assumptions FP.GuardedFragment)] indexCommuted fp8 p := by
  cases fp8 <;> equiv_decompose <;> fp_prove

variable {α : Type} [Inhabited α]

theorem row_product (M : Algebra α) (precision : Option ComputeDType) (s : State α) (n : Nat) :
    evalOp M precision (.mul .nat .nil (.ref .nat [] "pid_m") (.constNat n)) s =
      evalOp M precision (.mul .nat .nil (.constNat n) (.ref .nat [] "pid_m")) s := by
  cases h : s.regs .nat [] "pid_m" with
  | none => simp [evalOp_unfold,h]
  | some v =>
    simp only [evalOp_unfold,h]
    apply congrArg some
    funext i
    exact Nat.mul_comm _ _

theorem row_expression (M : Algebra α) (precision : Option ComputeDType) (s : State α) (n : Nat) :
    evalOp M precision (.add .nat .scalarL
      (.mul .nat .nil (.ref .nat [] "pid_m") (.constNat n)) (.arange n)) s =
    evalOp M precision (.add .nat .scalarL
      (.mul .nat .nil (.constNat n) (.ref .nat [] "pid_m")) (.arange n)) s := by
  change ((evalOp M precision (.mul .nat .nil (.ref .nat [] "pid_m") (.constNat n)) s).bind
      fun a => (evalOp M precision (.arange n) s).bind fun b => some (bop Nat.add .scalarL a b)) = _
  rw [row_product]
  rfl

theorem row_assignment (M : Algebra α) (s : State α) (n : Nat) :
    step M (.assign .nat [n] "offs_m" (.alg (.add .nat .scalarL
      (.mul .nat .nil (.ref .nat [] "pid_m") (.constNat n)) (.arange n)))) s =
    step M (.assign .nat [n] "offs_m" (.alg (.add .nat .scalarL
      (.mul .nat .nil (.constNat n) (.ref .nat [] "pid_m")) (.arange n)))) s := by
  simp only [step,evalExpr,evalComputeOp,row_expression]

def candidate (fp8 : Bool) (p : KernelSource.Parameters) :=
  TypedSource.indexCommutedKernel fp8 (KernelSource.arguments p)
def candidateStores (fp8 : Bool) (p : KernelSource.Parameters) :=
  (candidate fp8 p).surfaceBody.drop 17

theorem tail_run (M : Algebra α) (fp8 : Bool) (p : KernelSource.Parameters) (s : State α) :
    run M (KernelSource.stores fp8 p) s = run M (candidateStores fp8 p) s := by
  cases fp8 <;>
    simp only [KernelSource.stores,KernelSource.kernel,TypedSource.kernel,RmsMatmulRbe.TypedSource.kernel,candidateStores,
      candidate,TypedSource.indexCommutedKernel,KernelSource.arguments,
      ComputeKernel.surfaceBody,List.drop,run,row_assignment]

theorem tail_safe (M : Algebra α) (fp8 : Bool) (p : KernelSource.Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (KernelSource.stores fp8 p) s ↔
      Safety.traceSafeList M bounds (candidateStores fp8 p) s := by
  cases fp8 <;>
    simp only [KernelSource.stores,KernelSource.kernel,TypedSource.kernel,RmsMatmulRbe.TypedSource.kernel,candidateStores,
      candidate,TypedSource.indexCommutedKernel,KernelSource.arguments,
      ComputeKernel.surfaceBody,List.drop,Safety.traceSafeList_cons,Safety.traceSafeList_nil,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeAlg_unfold,
      Safety.safeOp_unfold,Safety.safeCompute_unfold,
      row_assignment]

/-- Exact full-state equality for every scalar backend and every input word. -/
theorem execution (M : Algebra α) (fp8 : Bool) (p : KernelSource.Parameters) (s : State α) :
    FP.Structural.exec M (KernelSource.kernel fp8 p) s = FP.Structural.exec M (candidate fp8 p) s := by
  have hbody : (candidate fp8 p).surfaceBody =
      (KernelSource.kernel fp8 p).surfaceBody.take 17 ++ candidateStores fp8 p := by
    cases fp8 <;> rfl
  change run M (KernelSource.kernel fp8 p).surfaceBody s = run M (candidate fp8 p).surfaceBody s
  rw [hbody]
  conv_lhs => rw [← List.take_append_drop 17 (KernelSource.kernel fp8 p).surfaceBody]
  rw [run_append,run_append]
  exact congrArg (fun f => (run M ((KernelSource.kernel fp8 p).surfaceBody.take 17) s).bind f)
    (funext (tail_run M fp8 p))

/-- The same concrete access trace is checked before and after the integer rewrite. -/
theorem trace_safe (M : Algebra α) (fp8 : Bool) (p : KernelSource.Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    TraceSafe M bounds (KernelSource.kernel fp8 p) s ↔ TraceSafe M bounds (candidate fp8 p) s := by
  have hbody : (candidate fp8 p).surfaceBody =
      (KernelSource.kernel fp8 p).surfaceBody.take 17 ++ candidateStores fp8 p := by
    cases fp8 <;> rfl
  unfold TraceSafe
  rw [hbody]
  conv_lhs => rw [← List.take_append_drop 17 (KernelSource.kernel fp8 p).surfaceBody]
  simp only [Safety.traceSafeList_append]
  exact and_congr_right (fun _ => forall_congr' (fun t => imp_congr_right (fun _ => tail_safe M fp8 p bounds t)))

#axiomsClean readable_preserves
#axiomsClean equivalent
#axiomsClean execution
#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.SourcePair
