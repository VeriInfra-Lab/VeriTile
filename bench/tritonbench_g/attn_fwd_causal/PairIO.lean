import bench.tritonbench_g.attn_fwd_causal.NativeIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.AttnFwdCausal.PairIO
open VeriTile Triton FP.Structural Source Whole
noncomputable section

def invocation (p : Parameters) (pid : Program) : Invocation Native.Expr :=
  { IO.launch Native.algebra Native.dotModel p pid with kernel := PairConnection.kernel p }

def candidate (instructions : Native.Instructions) (p : Parameters) (work : List Program) :
    ProgramIO.Variant (NativeIO.program instructions p work) where
  schedule := fun _ => work.map (invocation p)
  observe := (NativeIO.program instructions p work).observe

end
end VeriTile.Bench.Repairs.AttnFwdCausal.PairIO
