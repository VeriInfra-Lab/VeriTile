import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.Memory
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Mathematics

namespace VeriTile.Bench.Repairs.QuantizeCopyKv
set_option autoImplicit false
set_option maxHeartbeats 2000000
noncomputable section

def truncateReal (x : ℝ) : Int := if x < 0 then -Int.floor (-x) else Int.floor x
@[simp] theorem truncateReal_zero : truncateReal 0 = 0 := by simp [truncateReal]

def realOps : Operations ℝ ℝ where
  one := 1
  levels := 127
  abs := abs
  maximum := max
  divide := (· / ·)
  isZero := fun x => decide (x = 0)
  toHalf := id
  fromHalf := id
  int8 := fun x => VeriTile.Bench.Repairs.RowwiseQuantizationTriton.narrow8 (truncateReal x)

abbrev mathematicalMaximum := VeriTile.Bench.Repairs.RowwiseQuantizationTriton.logicalMaximum

/-- Independent logical-coordinate mathematical target; half casts are erased
only in this Real interpretation. No FP32-to-Real accuracy claim follows. -/
def mathematicalScale (p : Request) (x : Nat → ℝ) : ℝ := mathematicalMaximum x p.columns / 127
def mathematicalCode (p : Request) (x : Nat → ℝ) (c : Nat) : Byte :=
  if mathematicalMaximum x p.columns = 0 then VeriTile.Bench.Repairs.RowwiseQuantizationTriton.narrow8 0
  else VeriTile.Bench.Repairs.RowwiseQuantizationTriton.narrow8 (truncateReal (x c / mathematicalScale p x))

theorem real_maximum (p : Request) (d : Domain p) (plan : Plan p) (x : Nat → ℝ) :
    rowMaximum realOps plan.tree x = mathematicalMaximum x p.columns := by
  change plan.tree.evaluate max (fun c => |x c|) = mathematicalMaximum x p.columns
  apply le_antisymm
  · apply VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Tree.maximum_upper
    intro c hc
    exact VeriTile.Bench.Repairs.RowwiseQuantizationTriton.logicalMaximum_bound x p.columns c
      (List.mem_range.mp (plan.covers.mem_iff.mp hc))
  · apply VeriTile.Bench.Repairs.RowwiseQuantizationTriton.logicalMaximum_least
    · have hz := VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Tree.maximum_member plan.tree (fun c => |x c|) 0
        (plan.covers.mem_iff.mpr (List.mem_range.mpr d.columns_positive))
      exact le_trans (abs_nonneg _) hz
    · intro c hc
      exact VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Tree.maximum_member plan.tree (fun c => |x c|) c
        (plan.covers.mem_iff.mpr (List.mem_range.mpr hc))

theorem sourceRegisters_real (p : Request) (d : Domain p) (plan : Plan p) (x : Nat → ℝ) :
    (∀ c, c < p.columns → (sourceRegisters realOps plan.tree x).1 c = mathematicalCode p x c) ∧
    (sourceRegisters realOps plan.tree x).2 = mathematicalScale p x := by
  change (∀ c, c < p.columns → rowCode realOps (x c) (rowMaximum realOps plan.tree x) = mathematicalCode p x c) ∧
    rowScale realOps (rowMaximum realOps plan.tree x) = mathematicalScale p x
  rw [real_maximum p d plan x]
  constructor
  · intro c hc
    by_cases hm : mathematicalMaximum x p.columns = 0
    · have hx := (VeriTile.Bench.Repairs.RowwiseQuantizationTriton.logicalMaximum_zero x p.columns).mp hm c hc
      simp [rowCode,denominator,realOps,mathematicalCode,hm,hx]
    · simp [rowCode,denominator,rowScale,realOps,mathematicalCode,mathematicalScale,hm]
  · rfl

theorem real_zero_row (p : Request) (x : Nat → ℝ) (zero : ∀ c, c < p.columns → x c = 0) (c : Nat) :
    mathematicalScale p x = 0 ∧ mathematicalCode p x c = VeriTile.Bench.Repairs.RowwiseQuantizationTriton.narrow8 0 := by
  have hz := (VeriTile.Bench.Repairs.RowwiseQuantizationTriton.logicalMaximum_zero x p.columns).mpr zero
  simp [mathematicalScale,mathematicalCode,hz]

end
end VeriTile.Bench.Repairs.QuantizeCopyKv

#guard_msgs (drop info) in
#auditModuleAxioms
