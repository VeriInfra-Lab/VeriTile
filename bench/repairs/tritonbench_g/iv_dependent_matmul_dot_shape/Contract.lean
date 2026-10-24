import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Contract
import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.KernelDefect

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Contract
open TritonBenchG.IvDependentMatmul Whole

/-- Static FP32 dot operand shape is required only on the submitting path.
Empty output calls retain the original allocations and early return. -/
def DotShape (r : Request) : Prop := 0 < r.rows → 0 < r.columns → 16 ≤ r.blockInner
instance (r : Request) : Decidable (DotShape r) := inferInstanceAs (Decidable (_ → _ → _))

structure Admitted (r : Request) : Prop extends IvDependentMatmul.Contract.Admitted r where
  dotShape : DotShape r

def check (r : Request) (defaultFP32 cuda : Bool) : Except String Unit :=
  if DotShape r then IvDependentMatmul.Contract.check r defaultFP32 cuda else .error "dot shape"

def Preconditions (r : Request) (defaultFP32 cuda : Bool) : Prop :=
  defaultFP32 = Bool.true ∧ cuda = Bool.true ∧ Admitted r

theorem check_iff (r : Request) (defaultFP32 cuda : Bool) :
    check r defaultFP32 cuda = .ok () ↔ Preconditions r defaultFP32 cuda := by
  by_cases h : DotShape r
  · rw [check, if_pos h, IvDependentMatmul.Contract.check_iff]
    constructor
    · rintro ⟨hd, hc, ha⟩; exact ⟨hd, hc, ⟨ha, h⟩⟩
    · rintro ⟨hd, hc, ha⟩; exact ⟨hd, hc, ha.toAdmitted⟩
  · rw [check, if_neg h]
    constructor
    · intro impossible; cases impossible
    · rintro ⟨_, _, ha⟩; exact (h ha.dotShape).elim

theorem original (h : Preconditions r defaultFP32 cuda) :
    IvDependentMatmul.Contract.Preconditions r defaultFP32 cuda := ⟨h.1, h.2.1, h.2.2.toAdmitted⟩

theorem empty_preserved (r : Request) (empty : r.rows = 0 ∨ r.columns = 0)
    (defaultFP32 cuda : Bool) : check r defaultFP32 cuda = IvDependentMatmul.Contract.check r defaultFP32 cuda := by
  have : DotShape r := by rcases empty with h | h <;> simp [DotShape, h]
  exact if_pos this

theorem launched_shape (r : Request) (h : Admitted r) (hm : 0 < r.rows) (hn : 0 < r.columns) :
    VeriTile.Bench.Repairs.IvDependentMatmul.KernelDefect.FP32DotShape r :=
  ⟨r.positiveRowsBlock, r.positiveColumnsBlock, h.dotShape hm hn⟩

theorem original_counterexample_rejected
    (s : TritonBenchG.IvDependentMatmul.Kernels.Schedule) :
    check (VeriTile.Bench.Repairs.IvDependentMatmul.KernelDefect.request s) Bool.true Bool.true =
      .error "dot shape" := by rfl

#axiomsClean check_iff
#axiomsClean empty_preserved
#axiomsClean launched_shape
#axiomsClean original_counterexample_rejected
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Contract
