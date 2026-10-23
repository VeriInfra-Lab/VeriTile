import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.IOResults

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
noncomputable section

/-- Same complete two-phase source and input graph domain as the Real IO.
Output-only native evaluation preserves all input terms and cell tags. -/
abbrev nativeIO (BH T D BDF BDB : Nat) (U S V : Bool) :=
  IO.recurrentIO Native.sourceOps id BH T D BDF BDB U S V

/-- Independent ordered FP32 recurrence observed under the value-independent
compiler schedule. It preserves the original add/mul/FMA instruction boundary. -/
def nativeFormula (schedule : Native.Schedule) (BH T D : Nat) (U V : Bool)
    (vals : ChannelValues Native.Term (IO.inputs BH T D)) :
    ChannelValues (BitVec 32) (IO.outputs BH T D) :=
  let input := IO.asInput Native.sourceOps BH T D vals
  IO.result BH T D
    (fun bh d n => Native.eval Native.software schedule (state Native.sourceOps U input bh d n))
    (fun bh d t => Native.eval Native.software schedule (dx Native.sourceOps V input T bh d t))
    (fun bh d t => Native.eval Native.software schedule (dg Native.sourceOps U V input T bh d t))

theorem kernel_fp32 (p : Native.Instructions) (laws : Native.LocalConformance p)
    (schedule : Native.Schedule) (BH T D BDF BDB : Nat) (hF : 0<BDF) (hB : 0<BDB)
    (U S V : Bool) :
    (nativeIO BH T D BDF BDB U S V).Observes (fun _ _ => Native.eval p schedule)
      (nativeFormula schedule BH T D U V) :=
  IOResults.native p laws schedule BH T D BDF BDB hF hB U S V

#specNonCircular nativeFormula avoiding [forward,backward,Execution.gridRun,FP.Structural.exec]
#specNonCircular Native.LocalConformance avoiding [Execution.gridRun,FP.Structural.exec,state,dx,dg,Native.eval]
#axiomsClean kernel_fp32
#stmtSurfaceSubset kernel_fp32 ⊆
  [nativeIO,ProgramIO.Observes,nativeFormula,Native.Instructions,Native.LocalConformance,
   Native.Schedule,Native.eval,Native.Term,Native.instInhabitedTerm,
   IOChannel,MemoryIO.outputs,ProgramIO.toMemoryIO,TileIndex,IOChannel.shape,ProgramIO.mapOutputs._proof_1]
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
