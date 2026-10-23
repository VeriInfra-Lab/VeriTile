import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.IOImplements
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Scalar
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Gradient
import VeriTile.Triton.Float.ObservedProgramIO

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.IOResults
open VeriTile.Triton FP.Structural
noncomputable section

def nativeTarget (p : Native.Instructions) (schedule : Native.Schedule)
    (BH T D : Nat) (U V : Bool) (vals : ChannelValues Native.Term (IO.inputs BH T D)) :
    ChannelValues (BitVec 32) (IO.outputs BH T D) :=
  let input := IO.asInput Native.sourceOps BH T D vals
  IO.result BH T D
    (fun bh d n => Native.eval p schedule (state Native.sourceOps U input bh d n))
    (fun bh d t => Native.eval p schedule (dx Native.sourceOps V input T bh d t))
    (fun bh d t => Native.eval p schedule (dg Native.sourceOps U V input T bh d t))

/-- The shared observer changes only output payloads; arbitrary input graph
terms, actual phase execution, typed output cells and raw frame are retained. -/
theorem native (p : Native.Instructions) (laws : Native.LocalConformance p)
    (schedule : Native.Schedule) (BH T D BDF BDB : Nat) (hF : 0<BDF) (hB : 0<BDB)
    (U S V : Bool) :
    (IO.recurrentIO Native.sourceOps id BH T D BDF BDB U S V).Observes
      (fun _ _ => Native.eval p schedule) (nativeTarget Native.software schedule BH T D U V) := by
  rw [Native.software_eq p laws]
  have base := IOImplements.implements Native.sourceOps id BH T D BDF BDB hF hB U S V
  have observed := ProgramIO.Observes.of_implements base (fun _ _ => Native.eval Native.software schedule)
  have mapped : (fun vals => (IO.recurrentIO Native.sourceOps id BH T D BDF BDB U S V).mapOutputs
      (fun _ _ => Native.eval Native.software schedule)
      (IOImplements.target Native.sourceOps id BH T D U V vals))=
      nativeTarget Native.software schedule BH T D U V := by
    funext vals o j
    fin_cases o <;> rfl
  rw [mapped] at observed
  exact observed

#axiomsClean native
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.IOResults
