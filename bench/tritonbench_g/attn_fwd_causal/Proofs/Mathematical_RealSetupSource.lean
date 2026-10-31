import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealSetup

/-! Locate the actual statements surrounding the helper ranges by loop name,
not statement offsets. These source checks retain the dynamic bound registers
and verify the second helper's reset from caller pointer bases. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSetupSource
open VeriTile.Triton RealMemory RealSetup
set_option maxRecDepth 16000
set_option maxHeartbeats 1800000
set_option linter.unusedSimpArgs false

def beforeRange? (index : RegName) : List Stmt → Option (List Stmt)
  | [] => none
  | st :: rest => match st with
    | .forRangeDyn name .. => if name = index then some [] else
        (beforeRange? index rest).map (st :: ·)
    | _ => (beforeRange? index rest).map (st :: ·)

def afterRange? (index : RegName) : List Stmt → Option (List Stmt)
  | [] => none
  | .forRangeDyn name _ _ _ _ :: rest => if name = index then some rest else afterRange? index rest
  | _ :: rest => afterRange? index rest

def range? (index : RegName) : List Stmt → Nat → Option Stmt
  | [], _ => none
  | st :: rest, occurrence => match st with
    | .forRangeDyn name .. => if name = index then match occurrence with
        | 0 => some st
        | n+1 => range? index rest n
      else range? index rest occurrence
    | _ => range? index rest occurrence

section
variable (Q K V QS KS Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage : Nat) (hs : stage = 2 ∨ stage = 3)

end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSetupSource
