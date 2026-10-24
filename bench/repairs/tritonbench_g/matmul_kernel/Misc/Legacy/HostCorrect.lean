import bench.repairs.tritonbench_g.matmul_kernel.Contract
import bench.repairs.tritonbench_g.matmul_kernel.Misc.Whole
namespace VeriTile.Bench.Repairs.MatmulKernel.Legacy.HostCorrect
 theorem correct (c : Config) (h : Public.Admitted c) (m : Memory ℝ) :
    Public.RealClaim c m (Host.realCall c m) := Public.whole_real c h m
#axiomsClean correct
end VeriTile.Bench.Repairs.MatmulKernel.Legacy.HostCorrect

namespace VeriTile.Bench.Repairs.MatmulKernel.Checked
noncomputable section

def runReal (p : Metadata) (initial : Memory ℝ) : Except Error (Host.Result ℝ) := do
  checkInputs p
  return Host.realCall p.config initial

theorem checked_real (p : Metadata) (h : checkInputs p = .ok ())
    (a : Allocation p.config.A) (b : Allocation p.config.B) (c : Allocation p.config.C)
    (initial : Memory ℝ) :
    runReal p initial = .ok (Host.realCall p.config initial) ∧
    Public.RealClaim p.config initial (Host.realCall p.config initial) :=
  ⟨by simp [runReal,h], Public.whole_real p.config (admitted_domain p h a b c) initial⟩

#axiomsClean checked_real
end
end VeriTile.Bench.Repairs.MatmulKernel.Checked
