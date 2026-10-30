import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Kernels
import VeriTile.Triton.Source

/-! Proof-only decomposition of the complete selected source. -/
namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Source
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

def parts (p : Parameters) : VeriTile.Triton.Source.DynamicRange :=
  (VeriTile.Triton.Source.dynamicRange? (kernel p).surfaceBody "_i").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
def prelude (p : Parameters) := (parts p).before
def body (p : Parameters) := (parts p).body
def epilogue (p : Parameters) := (parts p).after

theorem source_decomposition (p : Parameters) :
    (kernel p).surfaceBody = prelude p ++
      [.forRangeDyn "_i" (.constNat 0) (.ref .nat [] "num_iters") (.constNat 1) (body p)] ++ epilogue p := by
  rcases p with ⟨A,B,O,M,N,K,TM,TN,TK,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk <;> rfl

#axiomsClean source_decomposition
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Source
