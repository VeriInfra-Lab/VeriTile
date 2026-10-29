import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.AddressIO
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.IONative

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPEquiv
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

def original (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Host.Call) (counts : Fin c.batch → Nat) (work : List Host.Job) :=
  ProgramIO.Implementation.original (PublicIO.nativeIO profiles ops c counts work) PublicIO.LengthsMatch

def earlyAddress (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Host.Call) (counts : Fin c.batch → Nat) (work : List Host.Job) :=
  ProgramIO.Implementation.ofVariant
    (AddressIO.variant (PublicIO.nativeModel profiles) NativeGraph.atom c
      (PublicIO.nativeObserve profiles ops c) counts work) PublicIO.LengthsMatch

/-- Exact output-address scheduling of the actual complete online source grid.
All original raw FP32 words, signed metadata, reduction/FMA/replica profiles,
lengths, workset permutations, output cells and exterior frames are retained. -/
specification reducev_grid_equiv (profiles : Host.Job → NativeGraph.Profile)
    (ops : NativeGraph.Instructions) (c : Host.Call) (counts : Fin c.batch → Nat)
    (work : List Host.Job) (complete : work.Perm (Host.workset c)) :
    original profiles ops c counts work ≡[[]] earlyAddress profiles ops c counts work := by
  exact ProgramIO.Implementation.of_equivalentOn
    (AddressIO.variant (PublicIO.nativeModel profiles) NativeGraph.atom c
      (PublicIO.nativeObserve profiles ops c) counts work)
    (AddressIO.equivalent (PublicIO.nativeModel profiles) NativeGraph.atom c
      (PublicIO.nativeObserve profiles ops c) counts work complete)

#axiomsClean reducev_grid_equiv
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPEquiv
