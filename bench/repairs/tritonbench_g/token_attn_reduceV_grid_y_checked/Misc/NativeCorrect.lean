import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.Proofs.NativeIO

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked
open VeriTile.Triton FP.Structural TokenAttnReduceV
noncomputable section

/-- Complete FP32 source grid. Symbolic input payloads retain exact source
addresses and typed-read defaults. All original word heaps specialize to this
IO; reducing inputs to leaves would incorrectly restrict the prior scope. -/
abbrev nativeIO := Interface.nativeIO

/-- Independent ordered recurrence with the existing reduction tree and
adjacent FMA selection, observed as FP32 words. Primitive instruction meanings
and compiler choices remain explicit; no numerical rewrite law is introduced. -/
abbrev floating := Interface.floating

theorem grid_native (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (p : Parameters) (batches heads : Nat) (counts : Fin batches → Nat)
    (work : List Grid.PID) (order : (Grid.fullGrid batches heads).Perm work)
    (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    (nativeIO R p batches heads counts work).ObservesOn
      (Interface.nativeObserver I S p batches heads)
      (Interface.LengthsMatch p batches heads counts) (floating I R S p batches heads counts) :=
  Interface.native_observes I R S p batches heads counts work order separate unique

#axiomsClean grid_native
#specNonCircular floating avoiding [kernel,FP.Structural.exec,Grid.run,outputValues]
#stmtSurfaceSubset grid_native ⊆
  [ProgramIO.ObservesOn,nativeIO,floating,Interface.nativeObserver,Interface.LengthsMatch,
   Native.Instructions,Native.Reductions,Native.Schedule,Native.Term,Native.instInhabitedTerm,Native.Word,
   Parameters,Grid.PID,Grid.fullGrid,Grid.Separate,Grid.Unique]
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked
