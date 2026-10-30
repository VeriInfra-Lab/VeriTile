import bench.tritonbench_g.attention_fwd_triton1.RealIO
import bench.tritonbench_g.attention_fwd_triton1.Proofs.IOInterface

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.RealIO
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

theorem reference_formula (c : Shape) (ifcond : Bool) (values : KernelIO.Values ℝ c) :
    KernelIO.reference realOps id c (KernelIO.realScale c) ifcond values=formula c values := by
  funext o i
  fin_cases o
  · have coord := KernelIO.main_coordinates c i.1.val i.1.isLt
    change Reference.output realOps c (KernelIO.logicalInputs c values) (KernelIO.realScale c)
      (KernelIO.mainHead c i.1.val) (KernelIO.mainRow c i.1.val/32) ifcond
      (KernelIO.mainRow c i.1.val%32) (KernelIO.mainColumn c i.1.val)=_
    exact RealProof.output_target c _ _ _ _ _ ifcond coord.2.1 coord.2.2
  · have coord := KernelIO.history_coordinates c i.1.val i.1.isLt
    change Reference.carry realOps c (KernelIO.logicalInputs c values) (KernelIO.realScale c)
      (KernelIO.historyHead c i.1.val) ifcond (KernelIO.historyChunk c i.1.val)
      (KernelIO.historyRow c i.1.val) (KernelIO.historyColumn c i.1.val)=_
    exact RealProof.carry_target c _ _ _ ifcond _ _ _ coord.2.2.1 coord.2.2.2

theorem implements (c : Shape) (storeFlag ifcond : Bool) :
    (io c storeFlag ifcond).Implements (formula c) := by
  have proof := KernelIO.implements realOps id rfl c (KernelIO.realScale c) storeFlag ifcond
  have equal : KernelIO.reference realOps id c (KernelIO.realScale c) ifcond=formula c :=
    funext (reference_formula c ifcond)
  rw [equal] at proof
  exact proof

#axiomsClean implements
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.RealIO
