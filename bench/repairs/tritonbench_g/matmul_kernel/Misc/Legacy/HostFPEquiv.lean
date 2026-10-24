import bench.repairs.tritonbench_g.matmul_kernel.Contract
import bench.repairs.tritonbench_g.matmul_kernel.Misc.Whole
namespace VeriTile.Bench.Repairs.MatmulKernel.Legacy.HostFPEquiv
 theorem fp_equiv (c : Config) (h : Public.Admitted c) (actual target : Native.Instructions)
    (ports : Native.Obligations actual target) (m : Physical.Bytes) :
    Public.NativeClaim c actual target m (Host.call c actual m) := Public.whole_fp c h actual target ports m
#axiomsClean fp_equiv
end VeriTile.Bench.Repairs.MatmulKernel.Legacy.HostFPEquiv

namespace VeriTile.Bench.Repairs.MatmulKernel.Checked

def runNative (p : Metadata) (instructions : Native.Instructions) (initial : Physical.Bytes) := do
  checkInputs p
  return Host.call p.config instructions initial

theorem checked_native (p : Metadata) (h : checkInputs p = .ok ())
    (a : Allocation p.config.A) (b : Allocation p.config.B) (c : Allocation p.config.C)
    (instructions : Native.Instructions) (initial : Physical.Bytes) :
    runNative p instructions initial = .ok (Host.call p.config instructions initial) ∧
    Public.NativeClaim p.config instructions instructions initial (Host.call p.config instructions initial) :=
  ⟨by simp [runNative,h], Public.whole_native p.config (admitted_domain p h a b c) instructions initial⟩

theorem checked_floating (p : Metadata) (h : checkInputs p = .ok ())
    (a : Allocation p.config.A) (b : Allocation p.config.B) (c : Allocation p.config.C)
    (actual target : Native.Instructions) (ports : Native.Obligations actual target) (initial : Physical.Bytes) :
    Public.NativeClaim p.config actual target initial (Host.call p.config actual initial) :=
  Public.whole_fp p.config (admitted_domain p h a b c) actual target ports initial

#axiomsClean checked_native
#axiomsClean checked_floating
end VeriTile.Bench.Repairs.MatmulKernel.Checked
