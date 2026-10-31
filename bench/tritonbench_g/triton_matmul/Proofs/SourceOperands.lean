import bench.tritonbench_g.triton_matmul.Proofs.SourceBodyExecution
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
noncomputable section

theorem aValue_signed {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (st : Storage) (s : State α) (k : Nat) (i : AIndex p) :
    (if (i.2.1.val : Int)<(p.K : Int)-(k : Int)*(BK p : Int) then
      read st (s.mem p.A (((aRow p s i.1 : Int)*(p.stride_am : Int)+
        ((if (i.2.1.val : Int)<(p.K : Int)-(k : Int)*(BK p : Int) then (k : Int)*(BK p : Int) else 0)+
         (if (i.2.1.val : Int)<(p.K : Int)-(k : Int)*(BK p : Int) then (i.2.1.val : Int) else 0))*
           (p.stride_ak : Int)).toNat))
    else M.cast none .real (dtype st) (M.literal none .real 0.0))=aValue M p st s k i := by
  by_cases h:i.2.1.val<p.K-k*BK p
  · have hi:(i.2.1.val : Int)<(p.K : Int)-(k : Int)*(BK p : Int):=by omega
    simp only [aValue,h,hi,↓reduceIte]
    simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    congr 2
    unfold aOffset
    ring
  · have hi:¬(i.2.1.val : Int)<(p.K : Int)-(k : Int)*(BK p : Int):=by omega
    simp only [h,hi,↓reduceIte,aValue]

theorem bValue_signed {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (st : Storage) (s : State α) (k : Nat) (i : BIndex p) :
    (if (i.1.val : Int)<(p.K : Int)-(k : Int)*(BK p : Int) then
      read st (s.mem p.B ((((if (i.1.val : Int)<(p.K : Int)-(k : Int)*(BK p : Int) then (k : Int)*(BK p : Int) else 0)+
         (if (i.1.val : Int)<(p.K : Int)-(k : Int)*(BK p : Int) then (i.1.val : Int) else 0))*
           (p.stride_bk : Int)+(bCol p s i.2.1 : Int)*(p.stride_bn : Int)).toNat))
    else M.cast none .real (dtype st) (M.literal none .real 0.0))=bValue M p st s k i := by
  by_cases h:i.1.val<p.K-k*BK p
  · have hi:(i.1.val : Int)<(p.K : Int)-(k : Int)*(BK p : Int):=by omega
    simp only [bValue,h,hi,↓reduceIte]
    simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    congr 2
    unfold bOffset
    ring
  · have hi:¬(i.1.val : Int)<(p.K : Int)-(k : Int)*(BK p : Int):=by omega
    simp only [h,hi,↓reduceIte,bValue]

#axiomsClean aValue_signed
#axiomsClean bValue_signed
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
