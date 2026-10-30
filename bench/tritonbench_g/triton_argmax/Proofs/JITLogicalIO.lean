import bench.tritonbench_g.triton_argmax.IO
import bench.tritonbench_g.triton_argmax.Proofs.JITRealPublicIO
import bench.tritonbench_g.triton_argmax.Proofs.JITRealDimensionIO

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section

theorem global_iff (kind : TypedScratch.Kind) (rest : Algebra Math.Value)
    (X V : RegionName) (I O : Region .int) (c : GlobalConfig) (index64 : Bool)
    (f : (Fin c.size → ℝ) → Int) :
    (logicalGlobal kind rest X V I O c index64).Implements f ↔
      (realGlobal kind rest X V I O c index64).Implements
        (fun xs => Fin.cases (fun _ => f (fun i => xs 0 (i,()))) (fun z => Fin.elim0 z)) := by rfl

theorem dimension_iff (rest : Algebra Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base pid : Nat) (index64 : Bool) (f : (Fin N → ℝ) → Int) :
    (logicalDimension rest X O M N K BM BN base pid index64).Implements
      (fun xs j => f (dimensionRow M N K BM base pid xs j)) ↔
    (realDimension rest X O M N K BM BN base pid index64).Implements
      (fun xs => Fin.cases (fun j => f (fun col : Fin N => JITRealDimensionIO.input M N K xs
        (((pid+base)/K)*BM+j.1.val) ((pid+base)%K) col.val)) (fun z => Fin.elim0 z)) := by
  have eq : (logicalDimension rest X O M N K BM BN base pid index64).lift
      (fun xs j => f (dimensionRow M N K BM base pid xs j)) =
      (fun xs => Fin.cases (fun j => f (fun col : Fin N => JITRealDimensionIO.input M N K xs
        (((pid+base)/K)*BM+j.1.val) ((pid+base)%K) col.val)) (fun z => Fin.elim0 z)) := by
    funext xs o j
    fin_cases o
    change f (dimensionRow M N K BM base pid (fun a b c => xs (0 : Fin 1) (a,b,c,())) j.1) =
      f (fun col : Fin N => JITRealDimensionIO.input M N K xs
        (((pid+base)/K)*BM+j.1.val) ((pid+base)%K) col.val)
    congr 1
    funext col
    by_cases hr : ((pid+base)/K)*BM+j.1.val < M <;>
      by_cases hk : (pid+base)%K < K <;>
      simp [dimensionRow,JITRealDimensionIO.input,hr,hk] <;> rfl
  exact congrArg (realDimension rest X O M N K BM BN base pid index64).Implements eq |>.to_iff
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITIO
