import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealPrologue

/-! Complete erased source decomposition for both typed kernels. This checks
every leading assignment, both helper setups/ranges, normalization and masked
store in their actual order. The decomposition uses named fragments. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSource
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def normalization (BM BD : Nat) (candidate : Bool) : Stmt :=
  .assign .real [BM,BD] "acc"
    (if candidate then
      .mul .real (.consSame (.consR .nil)) (.ref .real [BM,BD] "acc")
        (.div .real .scalarL (.const 1.0) (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "l_i")))
    else
      .div .real (.consSame (.consR .nil)) (.ref .real [BM,BD] "acc")
        (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "l_i")))

section
variable (Q K V QS KS Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage : Nat) (hs : stage = 2 ∨ stage = 3)

end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSource
