import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Execution

namespace VeriTile.Bench.Repairs.L2NormBwd.Profile
open Physical Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

def blocks : List Nat := [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]

theorem actual_block (cols : Nat) : blockSize cols∈blocks := by
  let k := Nat.clog 2 cols
  change min 16384 (2^k)∈blocks
  by_cases small : k<15
  · interval_cases k <;> norm_num [blocks]
  · have bound : 2^14≤2^k := Nat.pow_le_pow_right (by decide) (show 14≤k by omega)
    norm_num at bound
    rw [Nat.min_eq_left bound]
    norm_num [blocks]

theorem path_profile {M α β : Type} (ops : Ops M α β) (neg : Negation α) (red : Reduction α)
    (rt : Runtime M) (lib : Library ops neg rt) (call : Call) (eps : α) (initial after : World M)
    (output : View) (domain : InputDomain call initial)
    (path : WholePath ops red rt call eps initial output after) : path.context.body.layout.block∈blocks := by
  have norm := normalize_spec ops neg rt lib call initial domain
  rw [path.normalized] at norm
  change Normalized ops neg call initial path.request path.prepared at norm
  change path.request.block∈blocks
  rw [norm.blockActual]
  exact actual_block _

#axiomsClean actual_block
#axiomsClean path_profile
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Profile
