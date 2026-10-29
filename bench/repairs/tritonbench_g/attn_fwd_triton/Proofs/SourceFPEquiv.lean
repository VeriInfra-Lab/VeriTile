import bench.repairs.tritonbench_g.attn_fwd_triton.Kernels
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeGraphHost
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.TypedWitness
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.HostEmpty
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! The primary FP contract executes the frozen division candidate with its
selected native operation graph. Staged expressions explicitly represent the
two online-update FMA nodes, native score-scale operand order, initial constant
folds and same-dtype transports. FP16 probability/V casts and FP16 value-dot
result, FP32 normalization and BF16 output remain explicit. The complete
original word-valued invocation supplies the inputs to the reference graph;
physical output readback, ordinary host return, native addresses and full
caller frames are conclusions. No numerical rewrite law or domain is assumed.

The arbitrary source Algebra and reciprocal results below are auxiliary.
The latter consume an explicit div_mul_rcp model law; finite/nonzero does not
establish it for native FP32 (3/7 already refutes that inference). Neither
auxiliary identifies separately rounded arithmetic with native FMA. The
independent Real target remains in Correct.lean. Ordinary local native
instruction interpretation is separate from numerical accuracy to Real. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.FPEquivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

abbrev Rules := FP.ScalarArithmetic.Rules

section
variable (q k v qScale kScale out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL STAGE : Nat)
    (hStage : STAGE = 2 ∨ STAGE = 3)

/-- Check actual fp32 accumulators and row denominators after the full loop.
Every evaluated lane is finite and every row denominator is nonzero. This adds
no positivity, shape restriction, output equation or dot/reduction identity. -/
def divisionSite : Site :=
  .divBroadcast (Source.prefixBefore
    (TypedSource.repaired q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL STAGE hStage).surfaceBody "acc" 1)
    [BLOCK_M, BLOCK_DMODEL] BLOCK_M [1]
    (.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
    (.expandDim ⟨1, by simp⟩ (.ref .real [BLOCK_M] "l_i"))

def repaired : Program where
  kernel := TypedSource.repaired q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL STAGE hStage
  domain := [divisionSite q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL STAGE hStage]

def reciprocal : Program :=
  { repaired q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL STAGE hStage with
    kernel := TypedSource.reciprocal q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL STAGE hStage }

specification attention_reciprocal_equiv (R : Rules) :
    repaired q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL STAGE hStage ≡[R]
    reciprocal q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL STAGE hStage := by
  exact FPEquivProofs.attention_reciprocal_equiv q k v qScale kScale out
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
    Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL STAGE hStage R

end

open TypedExecution TypedHost
open VeriTile.Bench.Repairs.LayerNormFwd.TypedPhysical (Physical pull)
open Verified.PhysicalCells (Layout)
open Verified.HostSetup (Call calls beforeEmpty nonemptyChecks)
open HostShape (blockD)

specification whole_public_fp (ops : NativeGraph.Operations)
    (request : Request NativeGraph.Word) (a : Layout) (caller : Physical NativeGraph.Word a.toLayout)
    (ret : Prepared request a caller) (template : State NativeGraph.Word) (ok : Call → Bool)
    (hbefore : calls ok beforeEmpty=.ok ()) (hscales : calls ok nonemptyChecks=.ok ())
    (hcompile : ok .compileAndLaunch=Bool.true)
    (hne : ret.p.batches*ret.p.heads*ret.p.context*ret.p.headDim≠0)
    (hlaunch : NativeLaunch.CudaLaunchable (launchShape ret.p) (blockD ret.p.headDim)) :
    TypedHost.setup ret ok=.launch ret.returnedOutput rfl hne hlaunch ∧ NativeCertificate ret ∧
    (ret.p.context≤274877906816 ∧ ret.p.batches*ret.p.heads≤65535 ∧ blockD ret.p.headDim≤1048576) ∧
    ∃ final, ∃ graphPhysical : Physical NativeGraph.Expr a.toLayout, ∃ physical : Physical NativeGraph.Word a.toLayout,
      Nonempty (GridRun NativeGraph.backend ret.p (blockD ret.p.headDim) Bool.false
        (pull a.toLayout (NativeGraph.liftPhysical ret.initial) (NativeGraph.liftState template)) final) ∧
      pull a.toLayout graphPhysical final=final ∧
      physical=NativeGraph.evaluatePhysical ops graphPhysical ∧
      (∀ c : Coordinate ret.p, physical (outputCell a ret.p ret.base ret.views c)=
        NativeGraph.reference ops ret.p (blockD ret.p.headDim) a ret.initial template c) ∧
      (∀ cell : Fin a.capacity, ¬ret.preparationWrites cell →
        (∀ k, Window ret.p k.1 k.2 → a.address k≠cell.val) → physical cell=caller cell) ∧
      (∀ cell : Fin a.capacity, ret.original.owned cell → physical cell=caller cell) := by
  exact NativeGraph.successful_return ops request a caller ret template ok
    hbefore hscales hcompile hne hlaunch

/-- Auxiliary source-Algebra contract; not an identification of split FP32
rounding with the selected native FMA graph. -/
specification whole_public_source_fp {α : Type} [Inhabited α] (A : Algebra α)
    (request : Request α) (a : Layout) (caller : Physical α a.toLayout)
    (ret : Prepared request a caller) (template : State α) (ok : Call → Bool)
    (hbefore : calls ok beforeEmpty=.ok ()) (hscales : calls ok nonemptyChecks=.ok ())
    (hcompile : ok .compileAndLaunch=Bool.true)
    (hne : ret.p.batches*ret.p.heads*ret.p.context*ret.p.headDim≠0)
    (hlaunch : NativeLaunch.CudaLaunchable (launchShape ret.p) (blockD ret.p.headDim))
    (support : DotSupport A (blockD ret.p.headDim)) :
    setup ret ok=.launch ret.returnedOutput rfl hne hlaunch ∧ NativeCertificate ret ∧
    (ret.p.context≤274877906816 ∧ ret.p.batches*ret.p.heads≤65535 ∧ blockD ret.p.headDim≤1048576) ∧
    ∃ final, ∃ physical : Physical α a.toLayout,
      Nonempty (GridRun A ret.p (blockD ret.p.headDim) Bool.false (pull a.toLayout ret.initial template) final) ∧
      pull a.toLayout physical final=final ∧
      (∀ c : Coordinate ret.p, physical (outputCell a ret.p ret.base ret.views c)=
        reference A ret.p (blockD ret.p.headDim) support (pull a.toLayout ret.initial template) c) ∧
      (∀ cell : Fin a.capacity, ¬ret.preparationWrites cell →
        (∀ k, Window ret.p k.1 k.2 → a.address k≠cell.val) → physical cell=caller cell) ∧
      (∀ cell : Fin a.capacity, ret.original.owned cell → physical cell=caller cell) := by
  exact TypedHost.division_successful_return A request a caller ret template ok
    hbefore hscales hcompile hne hlaunch support

/-- Optional reciprocal optimization: the explicit model law is additional to
finite/nonzero site checks, and is not established for the native candidate. -/
specification whole_public_reciprocal_conditional {α : Type} [Inhabited α] (R : FPEquivProofs.Rules)
    (A : Algebra α) (D : FP.Guarded.Domain α) (hM : FP.Guarded.Models R.assumptions A D)
    (request : Request α) (a : Layout) (caller : Physical α a.toLayout)
    (ret : Prepared request a caller) (template : State α) (ok : Call → Bool)
    (hbefore : calls ok beforeEmpty=.ok ()) (hscales : calls ok nonemptyChecks=.ok ())
    (hcompile : ok .compileAndLaunch=Bool.true)
    (hne : ret.p.batches*ret.p.heads*ret.p.context*ret.p.headDim≠0)
    (hlaunch : NativeLaunch.CudaLaunchable (launchShape ret.p) (blockD ret.p.headDim))
    (support : DotSupport A (blockD ret.p.headDim))
    (hd : ∀ idx : ProgramIndex ret.p, ∀ site∈(program ret.p (blockD ret.p.headDim) Bool.false).domain,
      site.Holds A D (atProgram ret.p (pull a.toLayout ret.initial template) idx)) :
    setup ret ok=.launch ret.returnedOutput rfl hne hlaunch ∧ NativeCertificate ret ∧
    (ret.p.context≤274877906816 ∧ ret.p.batches*ret.p.heads≤65535 ∧ blockD ret.p.headDim≤1048576) ∧
    ∃ final, ∃ physical : Physical α a.toLayout,
      Nonempty (GridRun A ret.p (blockD ret.p.headDim) Bool.false (pull a.toLayout ret.initial template) final) ∧
      Nonempty (GridRun A ret.p (blockD ret.p.headDim) Bool.true (pull a.toLayout ret.initial template) final) ∧
      pull a.toLayout physical final=final ∧
      (∀ c : Coordinate ret.p, physical (outputCell a ret.p ret.base ret.views c)=
        reference A ret.p (blockD ret.p.headDim) support (pull a.toLayout ret.initial template) c) ∧
      (∀ cell : Fin a.capacity, ¬ret.preparationWrites cell →
        (∀ k, Window ret.p k.1 k.2 → a.address k≠cell.val) → physical cell=caller cell) ∧
      (∀ cell : Fin a.capacity, ret.original.owned cell → physical cell=caller cell) := by
  exact TypedHost.successful_return R A D hM request a caller ret template ok
    hbefore hscales hcompile hne hlaunch support hd

abbrev complete_source_success := @TypedExecution.success
abbrev complete_successful_equivalence := @TypedExecution.successful_equivalence
abbrev whole_physical_graph := @NativeGraph.successful_return
abbrev source_physical_grid := @TypedExecution.physical_division_grid
abbrev whole_physical_reciprocal_conditional := @TypedExecution.physical_grid_equivalence
abbrev original_input_coordinates := @TypedHost.initial_coordinates
abbrev exact_returned_output := @TypedHost.returned_output
abbrev empty_public_return := @HostEmpty.empty_return
abbrev infeasible_launch_failure := @TypedHost.infeasible_failure
abbrev ordered_first_failure := @Verified.HostSetup.first_failure
abbrev populated_normalization_domain := @TypedWitness.nonvacuous

#print_fp_assumptions attention_reciprocal_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.FPEquivalence
