import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.PairIO
import bench.tritonbench_g.softmax_flaggems.Proofs.FP

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourcePair
open VeriTile Triton FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000

theorem forward_noninner_equivalent (X Y : RegionName) (M N K B C : Nat) (one : Bool) :
    program (forward_noninner Y X M N K B C one) ≡[assumptions]
      program (AddressCandidate.forward_noninner Y X M N K B C one) := by
  equiv_decompose <;> fp_prove

theorem forward_inner_equivalent (X Y : RegionName) (M N B : Nat) (one : Bool) :
    program (forward_inner Y X M N B one) ≡[assumptions]
      program (AddressCandidate.forward_inner Y X M N B one) := by
  equiv_decompose <;> fp_prove

theorem backward_noninner_equivalent (P G Y : RegionName) (M N K B C : Nat) (one : Bool) :
    program (backward_noninner P G Y M N K B C one) ≡[assumptions]
      program (AddressCandidate.backward_noninner P G Y M N K B C one) := by
  equiv_decompose <;> fp_prove

theorem backward_inner_equivalent (P G Y : RegionName) (M N B C : Nat) (one : Bool) :
    program (backward_inner P G Y M N B C one) ≡[assumptions]
      program (AddressCandidate.backward_inner P G Y M N B C one) := by
  equiv_decompose <;> fp_prove

theorem forward_equivalent (X Y : RegionName) (d : Dimensions) (c : LaunchConfig) :
    forwardOriginal X Y d c ≡[assumptions] forwardAddressCommuted X Y d c := by
  unfold forwardOriginal forwardAddressCommuted
  split
  · exact forward_noninner_equivalent X Y d.M d.N d.K c.B c.C c.one
  · exact forward_inner_equivalent X Y d.M d.N c.B c.one

theorem backward_equivalent (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) :
    backwardOriginal P G Y d c ≡[assumptions] backwardAddressCommuted P G Y d c := by
  unfold backwardOriginal backwardAddressCommuted
  split
  · exact backward_noninner_equivalent P G Y d.M d.N d.K c.B c.C c.one
  · exact backward_inner_equivalent P G Y d.M d.N c.C c.B c.one

#axiomsClean forward_equivalent
#axiomsClean backward_equivalent
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourcePair
