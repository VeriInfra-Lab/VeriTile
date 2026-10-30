import bench.repairs.tritonbench_g.index_select_bwd.IO
import bench.repairs.tritonbench_g.index_select_bwd.Proofs.Interface

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Interface
open VeriTile.Triton FP.Structural
noncomputable section

/-- Equivalence of the complete slice judgment, including safety and frame. -/
theorem logical_iff [Inhabited A] (rest : Algebra A) (add : A → A → A) (p : Launch) :
    (logicalIO rest add p).Implements (fun xs i => add (xs.current i) (xs.incoming i)) ↔
      (programIO rest add p).Implements
        (fun xs => result (fun i => add (current xs i) (incoming xs i))) := Iff.rfl

theorem logical_implements [Inhabited A] (rest : Algebra A) (add : A → A → A)
    (p : Launch) (stride : p.stride = 1) (count : p.count ≤ p.block) :
    (logicalIO rest add p).Implements (fun xs i => add (xs.current i) (xs.incoming i)) :=
  (logical_iff rest add p).mpr (implements rest add p stride count)
end
end VeriTile.Bench.Repairs.IndexSelectBwd.Interface
