import bench.tritonbench_g.seeded_dropout.Proofs.KernelExecution
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.SeededDropout.KernelSafety
open VeriTile.Triton FP.Structural KernelExecution
open VeriTile.Bench.TritonBenchG.SeededDropout
open Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

private theorem prefix_safe {α : Type} [Inhabited α] (M : Algebra α) (c : Whole.Layout)
    (kind : Random.IntegerType) (p : ℝ) (seed : TypedSource.SeedCarrier kind)
    (pid : Nat) (s : State α) (bounds : RegionBounds)
    (input : ∀ j : Fin c.B,pid*c.B+j.val<c.n → pid*c.B+j.val<bounds c.X) :
    Safety.traceSafeList M bounds ((c.source kind p seed).surfaceBody.take 7) (tileState pid s) := by
  cases kind
  all_goals simp [Whole.Layout.source,TypedSource.source,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,numeric,natLt,bop,
    Region.cast,ofFloat,toFloat,tileState,State.setReg,Option.bind,Random.i32ArangeSize]
  all_goals exact input

private theorem suffix_safe {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (comparison : M.compareLt none .real = some lt) (c : Whole.Layout)
    (kind : Random.IntegerType) (p : ℝ) (seed : TypedSource.SeedCarrier kind)
    (pid : Nat) (s : State α) (bounds : RegionBounds)
    (hn : c.n<2^31) (hp : pid<SourceDomain.grid c.n c.B)
    (hs : kind.inRange (TypedSource.seedInt kind seed)=Bool.true)
    (output : ∀ j : Fin c.B,pid*c.B+j.val<c.n → pid*c.B+j.val<bounds c.OUT) :
    Safety.traceSafeList M bounds ((c.source kind p seed).surfaceBody.drop 7)
      (prefixState M c kind seed pid s) := by
  have random := prefix_rand M c kind seed pid s hn hp hs
  cases kind
  all_goals
    simp only [Whole.Layout.source,TypedSource.source,ComputeKernel.surfaceBody,
      List.drop_succ_cons,List.drop_zero,Random.i32ArangeSize,
      TypedExecution.seedOp,RandomDomain.profile] at random ⊢
    simp only [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,step,evalExpr,evalComputeOp]
    erw [random]
    simp [Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.safeOp_unfold,evalOp_unfold,store,
      Option.bind,ComputeDType.eraseDType,numeric,numericLt,ofFloat,toFloat,
      Region.cast,prefixState,State.setReg,comparison]
  all_goals exact output

/-- Actual source access safety, using the previously proved random-prefix
execution and the actual source successor states. RNG is not a safety axiom. -/
theorem trace_safe {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (comparison : M.compareLt none .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (s : State α) (hn : n<2^31) (hp : pid<gridCount n) (hs : SeedABI k seed)
    (bounds : RegionBounds)
    (input : ∀ j : Fin 1024,pid*1024+j.val<n → pid*1024+j.val<bounds X)
    (output : ∀ j : Fin 1024,pid*1024+j.val<n → pid*1024+j.val<bounds O) :
    TraceSafe M bounds (kernel k X O n p seed) (entry pid s) := by
  rw [source_eq]
  unfold TraceSafe
  rw [← List.take_append_drop 7 ((layout X O n).source (legacyKind k) p (legacySeed k seed)).surfaceBody,
    Safety.traceSafeList_append]
  refine ⟨prefix_safe M (layout X O n) (legacyKind k) p (legacySeed k seed) pid s bounds input,?_⟩
  intro t run
  rw [prefix_executes] at run
  cases run
  exact suffix_safe M lt comparison (layout X O n) (legacyKind k) p (legacySeed k seed)
    pid s bounds hn hp (seed_valid k seed hs) output

private def candidate (c : Whole.Layout) (kind : Random.IntegerType) (p : ℝ)
    (seed : TypedSource.SeedCarrier kind) : ComputeKernel :=
  TypedSource.source .fp32 kind Bool.true c.X c.OUT c.n p seed c.B c.legal

private theorem candidate_prefix_safe {α : Type} [Inhabited α] (M : Algebra α)
    (c : Whole.Layout) (kind : Random.IntegerType) (p : ℝ)
    (seed : TypedSource.SeedCarrier kind) (pid : Nat) (s : State α) (bounds : RegionBounds)
    (input : ∀ j : Fin c.B,pid*c.B+j.val<c.n → pid*c.B+j.val<bounds c.X) :
    Safety.traceSafeList M bounds ((candidate c kind p seed).surfaceBody.take 7) (tileState pid s) := by
  cases kind
  all_goals simp [candidate,TypedSource.source,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,numeric,natLt,bop,
    Region.cast,ofFloat,toFloat,tileState,State.setReg,Option.bind,Random.i32ArangeSize]
  all_goals simpa only [Nat.mul_comm] using input

private theorem candidate_prefix_executes {α : Type} [Inhabited α] (M : Algebra α)
    (c : Whole.Layout) (kind : Random.IntegerType) (p : ℝ)
    (seed : TypedSource.SeedCarrier kind) (pid : Nat) (s : State α) :
    run M ((candidate c kind p seed).surfaceBody.take 7) (tileState pid s) =
      some (prefixState M c kind seed pid s) := by
  have commuted : bop (fun x y : Nat => x*y) Broadcast.nil
      (fun _ : TileIndex [] => c.B) (fun _ => pid) =
      (fun _ : TileIndex [] => pid*c.B) := by
    funext i
    exact Nat.mul_comm c.B pid
  cases kind
  all_goals simp [candidate,TypedSource.source,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,numeric,natLt,numericLt,bop,
    Region.cast,Random.i32ArangeSize,prefixState,seedState,tileState,Nat.mul_comm]
  all_goals simp only [commuted]
  all_goals rfl

private theorem candidate_suffix (c : Whole.Layout) (kind : Random.IntegerType)
    (p : ℝ) (seed : TypedSource.SeedCarrier kind) :
    (candidate c kind p seed).surfaceBody.drop 7=(c.source kind p seed).surfaceBody.drop 7 := by
  cases kind <;> rfl

/-- The existing integer-product variant has the same checked source windows;
its prefix is proved to reach the same state before the retained RNG suffix. -/
theorem candidate_trace_safe {α : Type} [Inhabited α] (M : Algebra α)
    (lt : α → α → Bool) (comparison : M.compareLt none .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (s : State α) (hn : n<2^31) (hp : pid<gridCount n) (hs : SeedABI k seed)
    (bounds : RegionBounds)
    (input : ∀ j : Fin 1024,pid*1024+j.val<n → pid*1024+j.val<bounds X)
    (output : ∀ j : Fin 1024,pid*1024+j.val<n → pid*1024+j.val<bounds O) :
    TraceSafe M bounds (indexCommutedKernel k X O n p seed) (entry pid s) := by
  rw [candidate_eq]
  change Safety.traceSafeList M bounds
    (candidate (layout X O n) (legacyKind k) p (legacySeed k seed)).surfaceBody (tileState pid s)
  rw [← List.take_append_drop 7
      (candidate (layout X O n) (legacyKind k) p (legacySeed k seed)).surfaceBody,
    Safety.traceSafeList_append]
  refine ⟨candidate_prefix_safe M (layout X O n) (legacyKind k) p
    (legacySeed k seed) pid s bounds input,?_⟩
  intro t executed
  rw [candidate_prefix_executes] at executed
  cases executed
  rw [candidate_suffix]
  exact suffix_safe M lt comparison (layout X O n) (legacyKind k) p (legacySeed k seed)
    pid s bounds hn hp (seed_valid k seed hs) output

#axiomsClean candidate_trace_safe

#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.SeededDropout.KernelSafety
