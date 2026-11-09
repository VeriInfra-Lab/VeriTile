import bench.tritonbench_g.iv_dependent_matmul.Proofs.Sources.IvDependentMatmul_Contract_Contract

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Contract
open TritonBenchG.IvDependentMatmul Whole

abbrev DotShape := IvDependentMatmul.Contract.DotShape

/-- The compatibility interface carries the same current repaired conditions. -/
structure Admitted (r : Request) : Prop extends IvDependentMatmul.Contract.Admitted r

def check := IvDependentMatmul.Contract.check

def Preconditions (r : Request) (defaultFP32 cuda : Bool) : Prop :=
  defaultFP32 = Bool.true ∧ cuda = Bool.true ∧ Admitted r

theorem check_iff (r : Request) (defaultFP32 cuda : Bool) :
    check r defaultFP32 cuda = .ok () ↔ Preconditions r defaultFP32 cuda := by
  rw [check,IvDependentMatmul.Contract.check_iff]
  constructor
  · rintro ⟨hd,hc,ha⟩; exact ⟨hd,hc,⟨ha⟩⟩
  · rintro ⟨hd,hc,ha⟩; exact ⟨hd,hc,ha.toAdmitted⟩

theorem original (h : Preconditions r defaultFP32 cuda) :
    IvDependentMatmul.Contract.Preconditions r defaultFP32 cuda := ⟨h.1,h.2.1,h.2.2.toAdmitted⟩

theorem empty_preserved (r : Request) (_empty : r.rows = 0 ∨ r.columns = 0)
    (defaultFP32 cuda : Bool) : check r defaultFP32 cuda = IvDependentMatmul.Contract.check r defaultFP32 cuda := rfl

/-- Necessary FP32 frontend operand-shape condition on the actual launch path. -/
def FP32DotShape (r : Request) : Prop :=
  1 ≤ r.blockRows ∧ 1 ≤ r.blockColumns ∧ 16 ≤ r.blockInner

theorem launched_shape (r : Request) (h : Admitted r) (hm : 0 < r.rows) (hn : 0 < r.columns) :
    FP32DotShape r := ⟨r.positiveRowsBlock,r.positiveColumnsBlock,h.dotShape hm hn⟩

/-- Invalid nonempty one-lane dot tiles are rejected in every schedule. -/
theorem nonempty_small_block_rejected (s : Kernels.Schedule) :
    check (⟨"A","B","C",1,1,1,1,1,1,by decide,by decide,by decide,s,"cuda"⟩ : Request)
      Bool.true Bool.true = .error "dot shape" := by
  cases s <;> decide

#axiomsClean check_iff
#axiomsClean empty_preserved
#axiomsClean launched_shape
#axiomsClean nonempty_small_block_rejected
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Contract
