import bench.tritonbench_g.context_attn_llama.NativeIO
import bench.tritonbench_g.context_attn_llama.Proofs.NativeWindows
import bench.tritonbench_g.context_attn_llama.Proofs.NativeWholeSafety

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural NativeBindings
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

theorem initial_eq (c : Host.Call) (M : Algebra α) (idx : GridIndex c.grid) (s : State α) :
    (invocation c M idx).initial s = FPSuccess.atIndex s idx := rfl

theorem run_eq (c : Host.Call) (M : Algebra α) (order : List (GridIndex c.grid)) (s : State α) :
    runSchedule (order.map (invocation c M)) s = FPSuccess.runGrid c M order s := by
  induction order generalizing s with
  | nil => rfl
  | cons idx rest ih =>
    simp only [List.map_cons,runSchedule,FPSuccess.runGrid,Invocation.run,initial_eq]
    exact congrArg (Option.bind _) (funext ih)

theorem safe (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (order : List (GridIndex c.grid)) (s : State α) (metadata : FPSuccess.Metadata c s)
    (separate : c.regions.start≠c.regions.out ∧ c.regions.sequence≠c.regions.out ∧ c.regions.prompt≠c.regions.out)
    (bounds : RegionName → Nat)
    (reads : ∀ port off, off ∈ NativeArrays.reads c M dots order (snapshot c s) port →
      off < bounds (NativeArrays.ports c)[port].buf)
    (writes : ∀ idx ∈ order, ∀ i, c.active idx i → c.offset idx i < bounds c.regions.out) :
    ScheduleSafe bounds (order.map (invocation c M)) s := by
  have bn : 0<c.layout.BN := by rw [c.blockN]; exact Host.bm_positive c
  induction order generalizing s with
  | nil => trivial
  | cons idx rest ih =>
    change (invocation c M idx).Safe bounds s ∧ _
    refine ⟨?_,?_⟩
    · change TraceSafe (M.withDefaultPrecision .fp32) bounds
        (kernel c.layout c.regions .fp32 (Host.scale c.layout.D)) (FPSuccess.atIndex s idx)
      apply NativeSourceSafety.whole c.layout c.regions (Host.scale c.layout.D) M dots bn
      apply tile_bounds c idx s metadata bounds
      · intro port off member
        exact reads port off (List.mem_append_left _ member)
      · exact writes idx (List.mem_cons_self ..)
    · intro t ran
      obtain ⟨mid,executed,memory⟩ := FPReference.program c.layout c.regions
        (Host.scale c.layout.D) M dots bn (FPSuccess.atIndex s idx)
      change FP.Structural.exec (M.withDefaultPrecision .fp32)
        (kernel c.layout c.regions .fp32 (Host.scale c.layout.D)) (FPSuccess.atIndex s idx) = some t at ran
      rw [executed] at ran
      have same : t=mid := (Option.some.inj ran).symm
      subst t
      change mid.mem=(FPReference.tile c M dots idx s).mem at memory
      have nextMetadata : FPSuccess.Metadata c mid := by
        have h := tile_metadata c M dots idx s metadata separate
        simpa only [FPSuccess.Metadata,memory] using h
      have snap : snapshot c mid=NativeArrays.tile c M dots idx (snapshot c s) := by
        rw [←tile_snapshot c M dots idx s metadata]
        funext port off
        change (mid.mem _ _).read _ = ((FPReference.tile c M dots idx s).mem _ _).read _
        rw [memory]
      apply ih mid nextMetadata
      · intro port off member
        apply reads port off
        apply List.mem_append_right
        rw [←snap]
        exact member
      · intro job member
        exact writes job (List.mem_cons_of_mem _ member)

#axiomsClean run_eq
#axiomsClean safe
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeInterface
