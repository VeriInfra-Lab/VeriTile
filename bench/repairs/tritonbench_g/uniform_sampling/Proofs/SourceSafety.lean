import bench.repairs.tritonbench_g.uniform_sampling.Proofs.KernelExecution
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.UniformSampling.KernelSafety
open VeriTile.Triton FP.Structural KernelExecution
open VeriTile.Bench.TritonBenchG.UniformSampling
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000

private theorem list_safe_of_steps {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (code : List ComputeStmt)
    (safe : ∀ st ∈ code, ∀ s : State α, Safety.stepSafe M bounds st s)
    (s : State α) : Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil M bounds s
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) s)
    intro t _
    exact ih (fun e he => safe e (by simp [he])) t

/-- The first 23 actual source statements only manipulate registers. This
checks each source expression, without postulating a random or FMA law. -/
private theorem prefix_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (swapped : Bool) (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : Float32Bits) (B : Nat)
    (hB : Random.ArangeSizeDomain B) (s : State α) :
    Safety.traceSafeList M bounds
      ((KernelSource.source swapped y N seed offset lower upper B hB).surfaceBody.take 23) s := by
  apply list_safe_of_steps
  cases swapped
  all_goals
    simp only [KernelSource.source,Bool.false_eq_true,if_false,if_true,
      kernel,indexCommutedKernel,ComputeKernel.surfaceBody,List.take_succ_cons,List.take_zero,
      Random.i32ArangeSize]
    intro st member s
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
    all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,and_self]
    all_goals
      simp only [evalOp_unfold,if_true,true_and]
      apply list_safe_of_steps
      intro nested member state
      simp only [List.mem_cons,List.not_mem_nil,or_false] at member
      rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
      all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,
        Safety.safeOp_unfold,and_self]

private theorem store_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (y : RegionName) (N pid B : Nat) (f : Fin 4)
    (vn on : RegName) (s : State α)
    (ho : s.regs .int [B] on = some (KernelOffsets.offsets pid B f.val))
    (bound : ∀ i : Fin B, KernelOffsets.output pid B f.val i.val<N →
      KernelOffsets.output pid B f.val i.val<bounds y) :
    Safety.stepSafe M bounds (KernelStores.stmt y N B vn on) s := by
  simp [KernelStores.stmt,KernelStores.pointer,Safety.stepSafe_unfold,
    Safety.safeAddress_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeMask_unfold,Safety.safeOp_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,evalOp_unfold,ho,numeric,numericLt,bop,
    KernelOffsets.offsets,Option.bind,Region.cast]
  intro i active
  have h := bound i active
  cases y
  split <;> simp only [Int.toNat_natCast]
  · exact h
  · rename_i outside
    exact lt_of_le_of_lt (Nat.le_of_not_gt outside) h

private theorem stores_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (y : RegionName) (N pid B : Nat)
    (values : Fin 4 → Values α .real [B]) (s : State α)
    (hv : ∀ f,s.regs .real [B] (KernelStores.fieldName f)=some (values f))
    (ho : ∀ f,s.regs .int [B] (KernelStores.offsetName f)=some (KernelOffsets.offsets pid B f.val))
    (bound : ∀ f : Fin 4,∀ i : Fin B,KernelOffsets.output pid B f.val i.val<N →
      KernelOffsets.output pid B f.val i.val<bounds y) :
    Safety.traceSafeList M bounds (KernelStores.body y N B) s := by
  let v := fun f i => M.cast (some .fp32) .real .real (values f i)
  let s0 := KernelStores.field y N pid B 0 (v 0) s
  let s1 := KernelStores.field y N pid B 1 (v 1) s0
  let s2 := KernelStores.field y N pid B 2 (v 2) s1
  unfold KernelStores.body
  apply Safety.traceSafeList_cons_of_step
    (store_safe M bounds y N pid B 0 "r0" "off_0" s (ho 0) (bound 0))
    (KernelStores.step_exec M y N pid B 0 "r0" "off_0" (values 0) s (hv 0) (ho 0))
  apply Safety.traceSafeList_cons_of_step
    (store_safe M bounds y N pid B 1 "r1" "off_1" s0
      (by simpa [s0,KernelStores.offsetName] using ho 1) (bound 1))
    (KernelStores.step_exec M y N pid B 1 "r1" "off_1" (values 1) s0
      (by simpa [s0,KernelStores.fieldName] using hv 1)
      (by simpa [s0,KernelStores.offsetName] using ho 1))
  apply Safety.traceSafeList_cons_of_step
    (store_safe M bounds y N pid B 2 "r2" "off_2" s1
      (by simpa [s1,s0,KernelStores.offsetName] using ho 2) (bound 2))
    (KernelStores.step_exec M y N pid B 2 "r2" "off_2" (values 2) s1
      (by simpa [s1,s0,KernelStores.fieldName] using hv 2)
      (by simpa [s1,s0,KernelStores.offsetName] using ho 2))
  apply Safety.traceSafeList_cons_of_step
    (store_safe M bounds y N pid B 3 "r3" "off_3" s2
      (by simpa [s2,s1,s0,KernelStores.offsetName] using ho 3) (bound 3))
    (KernelStores.step_exec M y N pid B 3 "r3" "off_3" (values 3) s2
      (by simpa [s2,s1,s0,KernelStores.fieldName] using hv 3)
      (by simpa [s2,s1,s0,KernelStores.offsetName] using ho 3))
  exact Safety.traceSafeList_nil M bounds _

private theorem prefix_executes {α : Type} [Inhabited α] (M : Algebra α)
    (fused : α → α → α → α) (primitive : M.fma .fp32=some fused)
    (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (B : Nat) (hB : Random.ArangeSizeDomain B)
    (s : State α) (seedABI : Random.IntegerType.i64.inRange seed=Bool.true) :
    run M ((KernelSource.source swapped y N seed offset lower upper B hB).surfaceBody.take 23) s =
      some (offsetState M fused seed offset lower upper B s) := by
  have split : (KernelSource.source swapped y N seed offset lower upper B hB).surfaceBody.take 23 =
      ((KernelSource.source swapped y N seed offset lower upper B hB).surfaceBody.take 15 ++
        KernelAffine.body B) ++ KernelOffsets.body B := by cases swapped <;> rfl
  rw [split,run_append,run_append,
    KernelPrefix.prefix_exec M swapped y N seed offset lower upper B hB s seedABI,
    Option.bind_some,affine_exec M fused primitive,Option.bind_some,offset_exec]

/-- Actual source trace safety for both versions: four masked stores are checked
in their reached offset states, with every active pointer backed by the IO. -/
theorem trace_safe {α : Type} [Inhabited α] (M : Algebra α)
    (fused : α → α → α → α) (primitive : M.fma .fp32=some fused)
    (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (B : Nat) (hB : Random.ArangeSizeDomain B)
    (pid : Nat) (s : State α) (seedABI : Random.IntegerType.i64.inRange seed=Bool.true)
    (bounds : RegionName → Nat)
    (bound : ∀ f : Fin 4,∀ i : Fin B,KernelOffsets.output pid B f.val i.val<N →
      KernelOffsets.output pid B f.val i.val<bounds y) :
    TraceSafe M bounds (KernelSource.source swapped y N seed offset lower upper B hB) (entry pid s) := by
  unfold TraceSafe
  rw [← List.take_append_drop 23
    (KernelSource.source swapped y N seed offset lower upper B hB).surfaceBody,
    Safety.traceSafeList_append]
  refine ⟨prefix_safe M bounds swapped y N seed offset lower upper B hB _,?_⟩
  intro t executed
  rw [prefix_executes M fused primitive swapped y N seed offset lower upper B hB _ seedABI] at executed
  cases executed
  rw [KernelStores.source_slice]
  exact stores_safe M bounds y N pid B (values M fused seed offset lower upper B (entry pid s)) _
    (value_register M fused seed offset lower upper B (entry pid s))
    (offset_register M fused seed offset lower upper B (entry pid s)) bound

#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.UniformSampling.KernelSafety
