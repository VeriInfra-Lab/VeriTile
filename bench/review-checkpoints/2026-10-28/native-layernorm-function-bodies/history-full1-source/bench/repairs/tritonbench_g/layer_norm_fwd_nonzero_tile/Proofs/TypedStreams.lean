import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.TypedSource

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSupport
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

/-- Availability only. The callback is opaque and carries no order/Real law.
The source uses the FP32 real channel; the general signature keeps precision
resolution explicit while the proof evaluates those actual source requests. -/
structure Comparisons {α : Type} (A : Algebra α) where
  lt : Option ComputeDType → FloatDType → α → α → Bool
  available : ∀ p d, (engine A).compareLt p d = some (lt p d)

def firstStream (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) : List ComputeStmt :=
  ((actual f r stride M N eps p block hb).surfaceBody.drop 9).take 2

theorem firstStream_success {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s : State α) (hs : Ready f r stride block row s) :
    ∃ t, run (engine A) (firstStream f r stride M N eps p block hb) s = some t ∧
      Ready f r stride block row t := by
  by_cases hN : N = 1
  all_goals
    rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
    have hv := (Random.SourceProfile.inputsValid_iff _ _ _).mp
      (seed_valid ((s.mem r.seeds row).read .int) hb)
    have ho : ∀ j : Fin block, Random.IntegerType.i32.inRange (j.val : Int) = Bool.true :=
      fun j => hv.2 (j,PUnit.unit)
    have hu := hv.1
    simp at hu
    cases hsc : f.rowscale <;> cases hd : f.dropout <;> cases hm : f.dropoutMask
    all_goals simp [firstStream,actual,Source.kernel,Source.rawKernel,Source.rawKernelOne,hN,sourceFlags,sourceRegions,sourceStrides,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,integralMod,FP.Structural.bop,natLt,
      ←Int.natCast_mul,←Int.natCast_add,-Nat.cast_mul,-Nat.cast_add,
      numericLt,cmp.available,seed_valid,Region.cast,Random.i32ArangeSize,
      hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hu,ho,Ready,hsc,hd,hm]
    all_goals exact ⟨hx1,hres,hresout,hy1⟩

def secondInput (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) : List ComputeStmt :=
  ((actual f r stride M N eps p block hb).surfaceBody.drop 11).take 1

theorem secondInput_success {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s : State α) (hs : Ready f r stride block row s) :
    ∃ t, run (engine A) (secondInput f r stride M N eps p block hb) s = some t ∧
      Ready f r stride block row t := by
  by_cases hN : N = 1
  all_goals
    rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
    have hv := (Random.SourceProfile.inputsValid_iff _ _ _).mp
      (seed_valid ((s.mem r.seeds (M+row)).read .int) hb)
    have ho : ∀ j : Fin block, Random.IntegerType.i32.inRange (j.val : Int) = Bool.true :=
      fun j => hv.2 (j,PUnit.unit)
    have hu := hv.1
    simp at hu
    cases hxflag : f.x1 <;> cases hsc : f.rowscale <;> cases hd : f.dropout <;> cases hm : f.dropoutMask
    all_goals simp [secondInput,actual,Source.kernel,Source.rawKernel,Source.rawKernelOne,hN,sourceFlags,sourceRegions,sourceStrides,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,integralMod,FP.Structural.bop,natLt,
      ←Int.natCast_mul,←Int.natCast_add,-Nat.cast_mul,-Nat.cast_add,
      numericLt,cmp.available,seed_valid,Region.cast,Random.i32ArangeSize,
      hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hu,ho,Ready,hxflag,hsc,hd,hm]
    all_goals aesop

def residual (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) : List ComputeStmt :=
  ((actual f r stride M N eps p block hb).surfaceBody.drop 12).take 2

theorem residual_success {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s : State α) (hs : Ready f r stride block row s) :
    ∃ t, run (engine A) (residual f r stride M N eps p block hb) s = some t ∧
      Ready f r stride block row t := by
  by_cases hN : N = 1
  all_goals
    rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
    cases hrflag : f.residual <;> cases hoflag : f.residualOut
    all_goals simp [residual,actual,Source.kernel,Source.rawKernel,Source.rawKernelOne,hN,sourceFlags,sourceRegions,sourceStrides,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      ←Int.natCast_mul,←Int.natCast_add,-Nat.cast_mul,-Nat.cast_add,
      numericLt,Region.cast,Random.i32ArangeSize,
      hrow,hcols,hy,hx1,hres,hresout,hy1,hx,Ready,hrflag,hoflag]
    all_goals aesop

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
