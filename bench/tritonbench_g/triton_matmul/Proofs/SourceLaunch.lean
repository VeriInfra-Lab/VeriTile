import bench.tritonbench_g.triton_matmul.Proofs.Execution
import bench.tritonbench_g.triton_matmul.Proofs.Model
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.SourceLaunch
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul
open FPValue Repair
noncomputable section

theorem request_m (r : Request) : 0<BM r.parameters := by change 0<(128 : Nat); decide

theorem request_n (r : Request) : 0<BN r.parameters := by change 0<(256 : Nat); decide

theorem request_count (r : Request) : Source.ceilCount r.parameters.K (BK r.parameters)=r.parameters.numKBlocks := by
  apply SourceArithmetic.ceiling
  change 0<SourceScope.blockK r.storage
  cases r.storage <;> decide

theorem request_first {α : Type} (r : Request) (pid : Nat) (s : State α) (hp : pid<programs r) :
    SourceControl.first r.parameters (instanceState r pid s)≤SourceControl.rows r.parameters := by
  apply Nat.le_of_lt (SourceControl.first_bound _ _ ?_)
  have hs:=SourceControl.coordinates_eq r.parameters (instanceState r pid s) (request_m r) (request_n r)
  rw [hs.1,hs.2.1]
  simpa only [instanceState,↓reduceIte,Request.parameters,SourceScope.parameters,
    FPValue.blocksM,FPValue.blocksN,FPValue.BM,FPValue.BN,programs,tileRows,tileColumns,
    show (128 : Nat)-1=127 by decide,show (256 : Nat)-1=255 by decide,
    Nat.add_sub_assoc (by decide : 1≤(128 : Nat)),Nat.add_sub_assoc (by decide : 1≤(256 : Nat))] using hp

#axiomsClean request_count
#axiomsClean request_first
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.SourceLaunch
