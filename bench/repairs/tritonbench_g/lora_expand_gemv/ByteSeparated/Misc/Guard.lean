import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Contract

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.Contract
noncomputable section
open Classical

/-- Exact truth condition of the candidate Python overlap test. Python's
unbounded integer endpoint additions correspond to Nat addition here. -/
def pythonOverlaps (output read : StorageRange) : Bool :=
  decide (output.bytes≠0 ∧ read.bytes≠0 ∧ output.base<read.base+read.bytes ∧
    read.base<output.base+output.bytes)

theorem python_overlap_false_iff (output read : StorageRange) :
    pythonOverlaps output read=Bool.false ↔ output.disjoint read := by
  simp only [pythonOverlaps,decide_eq_false_iff_not,StorageRange.disjoint]
  omega

def pythonSeparate (a : Backings) : Bool :=
  !pythonOverlaps a.y a.x && !pythonOverlaps a.y a.w && !pythonOverlaps a.y a.indices

theorem python_separate_iff (a : Backings) : pythonSeparate a=Bool.true ↔ Separate a := by
  have flip (b : Bool) : Bool.not b=Bool.true ↔ b=Bool.false := by cases b <;> decide
  simp only [pythonSeparate,Bool.and_eq_true,flip,python_overlap_false_iff,Separate,and_assoc]

#axiomsClean python_overlap_false_iff
#axiomsClean python_separate_iff
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.Contract
