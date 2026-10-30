import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked
open VeriTile.Triton FP.Structural TokenAttnReduceV
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete source grid on named probability/value tensors. Counts match
runtime lengths; derived padding capacity adds no maximum-length guard.
Request, start and token metadata and all inactive entries are retained. -/
abbrev reductionIO := Logical.io

specification grid_real (p : Parameters) (batches heads : Nat) (counts : Fin batches → Nat)
    (work : List Grid.PID) (order : (Grid.fullGrid batches heads).Perm work)
    (positive : 0<p.block) (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    Spec.Real (reductionIO p batches heads counts work ⊨{Logical.domain p batches heads counts}
      fun xs b h d => ∑ n ∈ Finset.range (counts b), xs.probAt b h n * xs.valueAt b h n d) :=
  (Logical.judgment_iff p batches heads counts work).2
    (Interface.real_implements p batches heads counts work order positive separate unique)

#axiomsClean grid_real
#specNonCircular Logical.domain avoiding [kernel,FP.Structural.exec,Grid.run,outputValues]
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked
