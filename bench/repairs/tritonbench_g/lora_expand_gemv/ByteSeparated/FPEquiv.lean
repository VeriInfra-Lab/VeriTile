import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.Pair

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.FPEquiv
open VeriTile.Triton FP.Structural FPValue LoraIO
open FP.Equational (ReductionPlan)
open scoped VeriTile.Spec
noncomputable section

def original (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (raw : Nat → Int) (order : List Nat) :=
  ProgramIO.Implementation.original (io rest plans g order) (domain g raw) .exactCells

def unitSplit (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (raw : Nat → Int) (order : List Nat) :=
  ProgramIO.Implementation.ofVariant (Pair.variant rest plans g order) (domain g raw) .exactCells

/-- Complete actual kernel comparison. The selected wrapper passes SPLIT_N=N;
all invoked grids have N>0, so the candidate specializes cdiv(N,SPLIT_N) to 1.
FP32→FP16 conversion, half multiplication/reduction, optional FP32 addition,
all branches/masks, every CTA, successful safe execution, every output and both
complete exterior frames are retained, with no numerical rewrite assumption. -/
specification grid_equiv (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (raw : Nat → Int) (valid : Valid g raw) (order : List Nat)
    (coverage : order.Perm (List.range g.count)) :
    original rest plans g raw order ≡[[]] unitSplit rest plans g raw order :=
  ProgramIO.Implementation.of_equivalentOnCells (Pair.variant rest plans g order)
    (Pair.equivalent rest plans g raw valid order coverage)

#axiomsClean grid_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.FPEquiv
