import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Kernels
import bench.tritonbench_g.softmax_flaggems.Proofs.HostGridMetadata
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourcePair
open VeriTile Triton FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
noncomputable section

def program (source : ComputeKernel) : FP.ProfiledRewrite.Program := ⟨source, []⟩
def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forwardOriginal (X Y : RegionName) (d : Dimensions) (c : LaunchConfig) :=
  program (if 1<d.K then forward_noninner Y X d.M d.N d.K c.B c.C c.one
    else forward_inner Y X d.M d.N c.B c.one)

def forwardAddressCommuted (X Y : RegionName) (d : Dimensions) (c : LaunchConfig) :=
  program (if 1<d.K then AddressCandidate.forward_noninner Y X d.M d.N d.K c.B c.C c.one
    else AddressCandidate.forward_inner Y X d.M d.N c.B c.one)

def backwardOriginal (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) :=
  program (if 1<d.K then backward_noninner P G Y d.M d.N d.K c.B c.C c.one
    else backward_inner P G Y d.M d.N c.C c.B c.one)

def backwardAddressCommuted (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) :=
  program (if 1<d.K then AddressCandidate.backward_noninner P G Y d.M d.N d.K c.B c.C c.one
    else AddressCandidate.backward_inner P G Y d.M d.N c.C c.B c.one)

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourcePair
