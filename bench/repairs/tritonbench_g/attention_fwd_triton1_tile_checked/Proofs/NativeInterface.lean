import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.NativeIO
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.IOInterface

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.NativeIO
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section
set_option maxHeartbeats 800000

theorem reference_formula (native : ISA) (dom : NumericDomain) (c : Shape) (actualScale : Word)
    (ifcond : Bool) (values : KernelIO.Values Word c)
    (numeric : domain native dom c actualScale ifcond values) :
    KernelIO.reference (wordOps native) (literal actualScale) c (KernelIO.realScale c) ifcond values=
      formula c actualScale ifcond values := by
  funext o i
  fin_cases o
  · have coord := KernelIO.main_coordinates c i.1.val i.1.isLt
    change Reference.output (wordOps native) c (KernelIO.logicalInputs c values)
      (literal actualScale (KernelIO.realScale c)) (KernelIO.mainHead c i.1.val)
      (KernelIO.mainRow c i.1.val/32) ifcond (KernelIO.mainRow c i.1.val%32) (KernelIO.mainColumn c i.1.val)=_
    rw [literal_scale,←AttentionFwdTriton1.Whole001.Graph.eval_output]
    rw [NumericalReference.SourceFP.transport native dom numeric.scalarLaws _
      (numeric.admitted.output _ _ _ _ coord.1 (c.chunk_of_row _ coord.2.1)
        (Nat.mod_lt _ (by decide)) coord.2.2),AttentionFwdTriton1.Whole001.Graph.eval_output]
    rfl
  · have coord := KernelIO.history_coordinates c i.1.val i.1.isLt
    change Reference.carry (wordOps native) c (KernelIO.logicalInputs c values)
      (literal actualScale (KernelIO.realScale c)) (KernelIO.historyHead c i.1.val) ifcond
      (KernelIO.historyChunk c i.1.val) (KernelIO.historyRow c i.1.val) (KernelIO.historyColumn c i.1.val)=_
    rw [literal_scale,←AttentionFwdTriton1.Whole001.Graph.eval_carry]
    rw [NumericalReference.SourceFP.transport native dom numeric.scalarLaws _
      (numeric.admitted.carry _ _ _ _ coord.1 (Nat.le_of_lt coord.2.1)
        coord.2.2.1 coord.2.2.2),AttentionFwdTriton1.Whole001.Graph.eval_carry]
    rfl

theorem implements (native : ISA) (dom : NumericDomain) (c : Shape) (actualScale : Word)
    (storeFlag ifcond : Bool) :
    (io native c actualScale storeFlag ifcond).ImplementsOn (domain native dom c actualScale ifcond)
      (formula c actualScale ifcond) := by
  intro values numeric bounds bounded initial pinned
  have proof := KernelIO.implements (wordOps native) (literal actualScale) (by simp [wordOps])
    c (KernelIO.realScale c) storeFlag ifcond values trivial bounds bounded initial pinned
  rw [reference_formula native dom c actualScale ifcond values numeric] at proof
  exact proof

#axiomsClean implements
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.NativeIO
