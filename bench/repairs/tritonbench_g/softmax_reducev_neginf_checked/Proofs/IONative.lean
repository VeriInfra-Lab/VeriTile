import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.IOGrid

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
open VeriTile Triton FP.Structural
noncomputable section

def nativeFormula (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Host.Call) (counts : Fin c.batch → Nat) : Inputs NativeGraph.Word c counts → Outputs NativeGraph.Word c :=
  orderedTarget (nativeModel profiles) NativeGraph.atom c (nativeObserve profiles ops c) counts

theorem native_implements (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Host.Call) (counts : Fin c.batch → Nat) (work : List Host.Job)
    (complete : work.Perm (Host.workset c)) :
    (nativeIO profiles ops c counts work).ImplementsOn LengthsMatch (nativeFormula profiles ops c counts) :=
  ordered_implements (nativeModel profiles) NativeGraph.atom c (nativeObserve profiles ops c) counts work complete

#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
