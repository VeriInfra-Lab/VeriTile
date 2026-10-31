import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealFirstSetup

/-! Exact source connection for the complete middle sequence: local copies,
stage branch, first dynamic range, reset, and second dynamic range. Selectors
use the first_lo assignment and two start_n ranges, without numeric offsets. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMiddleSource
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 2200000
set_option linter.unusedSimpArgs false

def fromAssignment? (name : RegName) : List Stmt → Option (List Stmt)
  | [] => none
  | st :: rest => match st with
    | .assign _ _ target _ => if name = target then some (st :: rest) else fromAssignment? name rest
    | _ => fromAssignment? name rest

def throughRanges? (index : RegName) : List Stmt → Nat → Option (List Stmt)
  | _, 0 => some []
  | [], _+1 => none
  | st :: rest, n+1 => match st with
    | .forRangeDyn name .. =>
      (throughRanges? index rest (if name = index then n else n+1)).map (st :: ·)
    | _ => (throughRanges? index rest (n+1)).map (st :: ·)

section
variable (Q K V QS KS Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage : Nat) (hs : stage = 2 ∨ stage = 3)

end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMiddleSource
