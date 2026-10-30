import bench.tritonbench_g.attn_fwd_triton.Sources.Kernels
import VeriTile.Triton.Float.BroadcastRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! FP proof for the complete typed source and reciprocal candidate. Its
execution witness is provided by TypedWitness; whole-grid physical and host
composition is provided by TypedPhysical and TypedHost.
The prefix includes both staged attention loops, their exact causal
mask, casts, reductions and typed dot products. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

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

theorem attention_reciprocal_equiv (R : Rules) :
    repaired q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL STAGE hStage ≡[R]
    reciprocal q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL STAGE hStage := by
  equiv_decompose
  all_goals fp_prove

end

#print_fp_assumptions attention_reciprocal_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttnFwdTriton.FPEquivProofs
