import bench.tritonbench_g.context_attn_nopad.Sources.Mathematical_Kernels
import bench.tritonbench_g.context_attn_nopad.Sources.Mathematical_HostConfig

/-! Separate candidate bound in source-binding.json. Float operations are the
unchanged original recurrence: fp32 storage/work, source default dot precision,
natural tl.exp, causal -∞, fp32 reduction/casts, 128×128 tiles, 4/8 warps and one
stage. Python normalizes all five read roles and negative output; native integer
casts and Cartesian launch partitions are detailed in Proofs.NativeIndex.
The floating body is shared unchanged. NativeProgram supplies the checked
BitVec64 projection of every changed integer source site; HostPlan supplies
metadata synchronization and the exact split-grid index family. -/
namespace VeriTile.Bench.Repairs.ContextAttnNopad
open VeriTile.Triton
abbrev Call := VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host.Call
abbrev Regions := VeriTile.Bench.TritonBenchG.ContextAttnNopad.Program.Regions
noncomputable def kernel (c : Call) (r : Regions) : ComputeKernel :=
  c.source r Bool.false

theorem source_reuse (c : Call) (r : Regions) :
    kernel c r = c.source r Bool.false := rfl

def originalSHA256 : String :=
  "5135592994e22667902175fca73221ce017294706324ef83d73110290dd1d869"
end VeriTile.Bench.Repairs.ContextAttnNopad
