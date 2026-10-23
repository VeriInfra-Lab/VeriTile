import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.NativeHost
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Integer
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
set_option maxHeartbeats 4000000

inductive DataRole where | query | key | value | output deriving DecidableEq

def DataRole.region (d : DataRole) (c : Host.Call) := match d with
  | .query => c.regions.query | .key => c.regions.key | .value => c.regions.value | .output => c.regions.out
def DataRole.row (d : DataRole) (c : Host.Call) := match d with
  | .query => c.layout.qRow | .key => c.layout.kRow | .value => c.layout.vRow | .output => c.layout.outRow
def DataRole.head (d : DataRole) (c : Host.Call) := match d with
  | .query => c.layout.qHead | .key => c.layout.kHead | .value => c.layout.vHead | .output => c.layout.outHead
def DataRole.dim (d : DataRole) (c : Host.Call) := match d with
  | .query => c.layout.qDim | .key => c.layout.kDim | .value => c.layout.vDim | .output => c.layout.outDim

def qOffset (c : Host.Call) (role : DataRole) (idx : GridIndex c.grid) (i d : Nat) :=
  (c.start (batchAt c idx)+Formula.queryRow c.layout (c.launchAt idx) i)*role.row c+
    Formula.head c.layout (c.launchAt idx)*role.head c+d*role.dim c

def kvOffset (c : Host.Call) (role : DataRole) (idx : GridIndex c.grid) (step j d : Nat) :=
  mathPage c idx step j*role.row c+(Formula.head c.layout (c.launchAt idx)/c.layout.kvGroup)*role.head c+d*role.dim c

def qOffsetBits (c : Host.Call) (w : Widths) (role : DataRole) (idx : GridIndex c.grid) (i d : Nat) :=
  queryBits c.layout (c.launchAt idx) w.start (c.start (batchAt c idx)) i d (role.row c) (role.head c) (role.dim c)
def kvOffsetBits (c : Host.Call) (w : Widths) (role : DataRole) (idx : GridIndex c.grid) (step j d : Nat) :=
  valueBits c.layout (c.launchAt idx) (pageBits c w idx step j) d (role.row c) (role.head c) (role.dim c)

end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Integer
