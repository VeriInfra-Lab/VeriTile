import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.TypedStreams

namespace VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open TypedExecution
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def normalization (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) : List ComputeStmt :=
  ((repaired f r stride M N eps p block hb).surfaceBody.drop 15).take 3

def NormalizedReady {α : Type} (f : Flags) (r : Regions) (stride : Strides)
    (block row : Nat) (s : State α) : Prop :=
  Ready f r stride block row s ∧ (∃ std, s.regs .real [] "rstd" = some std) ∧
    (f.rms=Bool.false → ∃ mean, s.regs .real [] "mean" = some mean)

theorem normalization_success {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s : State α) (hs : Ready f r stride block row s) :
    ∃ t, run (engine A) (normalization f r stride M N eps p block hb) s = some t ∧
      NormalizedReady f r stride block row t := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
  cases hrms : f.rms
  all_goals simp [normalization,repaired,original,ComputeKernel.surfaceBody,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
    numericLt,Region.cast,Random.i32ArangeSize,TileShape.axisDim,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,NormalizedReady,Ready,hrms]
  all_goals aesop

def affine (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) : List ComputeStmt :=
  (repaired f r stride M N eps p block hb).surfaceBody.drop 18

theorem affine_success {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s : State α) (hs : NormalizedReady f r stride block row s) :
    ∃ t, run (engine A) (affine f r stride M N eps p block hb) s = some t := by
  rcases hs with ⟨⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩,⟨std,hstd⟩,hm⟩
  cases hrms : f.rms
  · obtain ⟨mean,hmean⟩ := hm hrms
    cases hbias : f.bias <;> cases hw1 : f.w1 <;> cases hb1 : f.b1
    all_goals simp [affine,repaired,original,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      numericLt,Region.cast,Random.i32ArangeSize,hcols,hy,hy1,hx,hstd,hmean,hrms,hbias,hw1,hb1]
  · cases hbias : f.bias <;> cases hw1 : f.w1 <;> cases hb1 : f.b1
    all_goals simp [affine,repaired,original,ComputeKernel.surfaceBody,
      run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,
      numericLt,Region.cast,Random.i32ArangeSize,hcols,hy,hy1,hx,hstd,hrms,hbias,hw1,hb1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
