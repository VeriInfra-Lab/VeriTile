import bench.repairs.tritonbench_g.rmsnorm_fused_llama.IO
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.NativeIO
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.RuntimeReference

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.Reference
open VeriTile Triton FP.Structural
noncomputable section

/-- Total finite-channel extensions; active output formulas never use the
fallback. Inactive source reduction lanes still use the source's own zero. -/
def rowInput (xs : Values Native.Term .real [rows,N]) (row : Fin rows) (k : Nat) : Native.Term :=
  if h : k<N then xs (row,⟨k,h⟩,PUnit.unit) else .literal none .real 0
def weightInput (ws : Values Native.Term .real [N]) (k : Nat) : Native.Term :=
  if h : k<N then ws (⟨k,h⟩,PUnit.unit) else .literal none .real 0

/-- Independent ordered native graph, including runtime epsilon and FP16
round trip; inputs remain arbitrary original symbolic terms. -/
def native (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (X Y W : RegionName) (rows N : Nat) (epsilon : Native.Word)
    (a : (NativeIO.original R X Y W rows N epsilon).Inputs) :
    (NativeIO.original R X Y W rows N epsilon).ObservedOutputs Native.Word :=
  fun o => Fin.cases (fun j => RuntimeReference.native I R S
    (rowInput (a (0 : Fin 2)) j.1) (weightInput (a (1 : Fin 2)))
    N 16384 (.leaf "epsilon" 0 epsilon) j.2.1.val) (fun i => Fin.elim0 i) o


#specNonCircular RuntimeReference.native avoiding [runtimeKernel,FP.Structural.exec,RuntimeRow.value,RuntimeGrid.graph]

end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.Reference
