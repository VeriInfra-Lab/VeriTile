import bench.repairs.tritonbench_g.int8_quantization.Proofs.RealTotal
import bench.repairs.tritonbench_g.int8_quantization.Proofs.TensorBinding

namespace VeriTile.Bench.Repairs.Int8Quantization.RealHost
open VeriTile.Triton FP.Structural Revision002 Words Direct
noncomputable section

theorem after_frame (c : Request) (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ)
    (before : Heap) (out : Allocation c before) (a : Nat) (outside : Outside c out a) :
    (after c rest store32 storage before out).memory a=before.memory a :=
  Bytes.store_frame _ _ _ _ _ outside.1 outside.2

theorem pair_frame (q k : Request) (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ)
    (before : Heap) (qa : Allocation q before)
    (ka : Allocation k (after q rest store32 storage before qa))
    (a : Nat) (outsideQ : Outside q qa a) (outsideK : Outside k ka a) :
    (after k rest store32 storage (after q rest store32 storage before qa) ka).memory a=before.memory a :=
  (after_frame k rest store32 storage _ ka a outsideK).trans
    (after_frame q rest store32 storage before qa a outsideQ)

#axiomsClean pair_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8Quantization.RealHost
