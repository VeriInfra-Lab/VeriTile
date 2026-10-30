import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.ScalarPair
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.ScalarSafety
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Pair
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.OptionalState

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Runtime
open VeriTile Triton FP.Structural FP.ProfiledRewrite FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable
variable {α : Type} [Inhabited α]
set_option maxRecDepth 32000
set_option maxHeartbeats 4000000

theorem classify_fits (floatValue : BitVec 64 → α) (raw : PythonNumber)
    (h : ScalarABI raw) : (classify floatValue raw).Fits := by
  cases raw with
  | «float» bits => trivial
  | integer value =>
    simp only [classify]
    split_ifs with one small signed
    · trivial
    · exact small
    · exact ⟨h.1,signed⟩
    · change value.toNat < 2^64
      change -(2^63 : Int) ≤ value ∧ value < 2^64 at h
      omega

/-- No new host restriction: the existing contract implies each live profile fits. -/
theorem admitted_profile (floatValue : BitVec 64 → α) (c : Shape)
    (xs : Inputs) (raw : PythonNumber) (h : InputContract c xs raw)
    (live : Launches c) : (classify floatValue raw).Fits :=
  classify_fits floatValue raw (h.2 live)

theorem source_exec (M : Algebra α) (c : Shape) (a : Scalar α) (s : State α)
    (hs : s.regs .real [] "scale"=some (fun _ => a.value M)) :
    exec (engine M) (source c a) s=exec (engine M) (kernel c) s := by
  cases a with
  | fp32 value => rfl
  | i32 value => exact Source.i32_source_eq M c s value hs
  | i64 value => exact Source.i64_source_eq M c s value hs
  | u64 value => exact Source.u64_source_eq M c s value hs
  | one => exact Source.one_source_eq M c s hs

theorem source_safe (M : Algebra α) (c : Shape) (a : Scalar α) (s : State α)
    (hs : s.regs .real [] "scale"=some (fun _ => a.value M)) (bounds : RegionName → Nat) :
    TraceSafe (engine M) bounds (source c a) s ↔ TraceSafe (engine M) bounds (kernel c) s := by
  cases a with
  | fp32 value => rfl
  | i32 value => exact Source.i32_source_safe M c s value hs bounds
  | i64 value => exact Source.i64_source_safe M c s value hs bounds
  | u64 value => exact Source.u64_source_safe M c s value hs bounds
  | one => exact Source.one_source_safe M c s hs bounds

theorem candidate_exec (M : Algebra α) (c : Shape) (a : Scalar α) (s : State α) :
    exec M (candidate c a) s=exec M (source c a) s := by
  cases a with
  | fp32 value => exact Pair.kernel_run M c s
  | i32 value => exact Pair.i32_kernel_run M c value s
  | i64 value => exact Pair.i64_kernel_run M c value s
  | u64 value => exact Pair.u64_kernel_run M c value s
  | one => exact Pair.one_kernel_run M c s

theorem candidate_safe (M : Algebra α) (c : Shape) (a : Scalar α) (s : State α)
    (bounds : RegionName → Nat) :
    TraceSafe M bounds (candidate c a) s ↔ TraceSafe M bounds (source c a) s := by
  cases a with
  | fp32 value => exact Pair.kernel_safe M c bounds s
  | i32 value => exact Pair.i32_kernel_safe M c value bounds s
  | i64 value => exact Pair.i64_kernel_safe M c value bounds s
  | u64 value => exact Pair.u64_kernel_safe M c value bounds s
  | one => exact Pair.one_kernel_safe M c bounds s

theorem invocation_run (M : Algebra α) (c : Shape) (a : Scalar α) (early : Bool)
    (p : FusedRwkv6Kernel.Program) (s : State α) :
    (invocation M c a early p).run s=(Interface.invocation M c (a.value M) p).run s := by
  have hs := Interface.Execution.invocation_scalar M c (a.value M) p s
  cases early
  · exact source_exec M c a _ hs
  · exact (candidate_exec (engine M) c a _).trans (source_exec M c a _ hs)

theorem invocation_safe (M : Algebra α) (c : Shape) (a : Scalar α) (early : Bool)
    (p : FusedRwkv6Kernel.Program) (bounds : RegionName → Nat) (s : State α) :
    (invocation M c a early p).Safe bounds s ↔
      (Interface.invocation M c (a.value M) p).Safe bounds s := by
  have hs := Interface.Execution.invocation_scalar M c (a.value M) p s
  cases early
  · exact source_safe M c a _ hs bounds
  · exact (candidate_safe (engine M) c a _ bounds).trans (source_safe M c a _ hs bounds)

theorem mapped_run (M : Algebra α) (c : Shape) (a : Scalar α) (early : Bool)
    (work : List FusedRwkv6Kernel.Program) (s : State α) :
    runSchedule (work.map (invocation M c a early)) s=
      runSchedule (work.map (Interface.invocation M c (a.value M))) s := by
  induction work generalizing s with
  | nil => rfl
  | cons p rest ih =>
    simpa only [List.map_cons,runSchedule,invocation_run] using
      congrArg (Option.bind ((Interface.invocation M c (a.value M) p).run s)) (funext ih)

theorem mapped_safe (M : Algebra α) (c : Shape) (a : Scalar α) (early : Bool)
    (work : List FusedRwkv6Kernel.Program) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (work.map (invocation M c a early)) s ↔
      ScheduleSafe bounds (work.map (Interface.invocation M c (a.value M))) s := by
  induction work generalizing s with
  | nil => rfl
  | cons p rest ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_safe,invocation_run] at *
    simp only [ih]

theorem schedule_run (M : Algebra α) (c : Shape) (a : Scalar α) (early : Bool)
    (xs : Interface.Inputs α c) (s : State α) :
    runSchedule (schedule M c a early) s=
      runSchedule ((Interface.programIO M c (a.value M)).schedule xs) s := by
  by_cases live : Launches c
  · simpa only [schedule,Interface.programIO,if_pos live] using mapped_run M c a early (programs c) s
  · simp only [schedule,Interface.programIO,if_neg live]

theorem schedule_safe (M : Algebra α) (c : Shape) (a : Scalar α) (early : Bool)
    (xs : Interface.Inputs α c) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (schedule M c a early) s ↔
      ScheduleSafe bounds ((Interface.programIO M c (a.value M)).schedule xs) s := by
  by_cases live : Launches c
  · simpa only [schedule,Interface.programIO,if_pos live] using
      mapped_safe M c a early (programs c) bounds s
  · simp only [schedule,Interface.programIO,if_neg live]

/-- The actual raw ABI bodies preserve the entire implementation judgment,
including success, safety, active typed outputs and the exterior cell frame. -/
theorem implements_iff (M : Algebra α) (c : Shape) (a : Scalar α)
    (f : Interface.Inputs α c → ChannelValues α (Interface.outputPorts c)) :
    (programIO M c a).Implements f ↔ (Interface.programIO M c (a.value M)).Implements f := by
  constructor
  · intro h xs hx bounds hb s hin
    obtain ⟨safe,t,ran,outputs,frame⟩ := h xs hx bounds hb s hin
    exact ⟨(schedule_safe M c a Bool.false xs bounds s).mp safe,t,
      (schedule_run M c a Bool.false xs s).symm.trans ran,outputs,frame⟩
  · intro h xs hx bounds hb s hin
    obtain ⟨safe,t,ran,outputs,frame⟩ := h xs hx bounds hb s hin
    exact ⟨(schedule_safe M c a Bool.false xs bounds s).mpr safe,t,
      (schedule_run M c a Bool.false xs s).trans ran,outputs,frame⟩

theorem equivalent (M : Algebra α) (c : Shape) (hc : Sizes c) (plan : Plans c)
    (reduction : Source.ReductionMatches M c plan) (a : Scalar α) :
    (programIO M c a).Equivalent (variant M c a) := by
  intro xs _ bounds bounded s binding
  obtain ⟨safe,result,ran,outputs,frame⟩ :=
    Interface.Proof.implements M c hc plan reduction (a.value M) xs trivial bounds bounded s binding
  refine ⟨(schedule_safe M c a Bool.false xs bounds s).mpr safe,
    (schedule_safe M c a Bool.true xs bounds s).mpr safe,result,result,
    (schedule_run M c a Bool.false xs s).trans ran,
    (schedule_run M c a Bool.true xs s).trans ran,?_,frame,frame⟩
  intro o j active
  rfl

/-- Absent H0/HT have zero accessible capacity in every scalar ABI profile. -/
theorem absent_channels_safe (M : Algebra α) (c : Shape) (a : Scalar α) (early : Bool)
    (s : State α) (hs : s.regs .real [] "scale"=some (fun _ => a.value M))
    (bounds : RegionName → Nat) (hb : SourceSafety.Bounds c s bounds) :
    TraceSafe (engine M) (SourceSafety.optionalBounds c bounds)
      (if early then candidate c a else source c a) s := by
  have safe := (source_safe M c a s hs _).mpr
    (SourceSafety.absent_channels_safe M c s (a.value M) hs bounds hb)
  cases early
  · exact safe
  · exact (candidate_safe (engine M) c a s _).mpr safe

#axiomsClean classify_fits
#axiomsClean admitted_profile
#axiomsClean source_exec
#axiomsClean candidate_exec
#axiomsClean implements_iff
#axiomsClean equivalent
#axiomsClean absent_channels_safe
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Runtime
