import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ProfileWitness

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
noncomputable section

/-- Byte extents inspected by TensorIterator's get_dim_to_split for a
contiguous [count,rows] FP32 input reduced along its first dimension. A
reduction-axis slice retains the original row stride. -/
def reductionExtent (count rows : Nat) : Nat := (count-1)*(rows*4)
def outputExtent (rows : Nat) : Nat := (rows-1)*4

/-- Before any output-axis split, a remaining reduction slice with at least
two inputs has strictly larger byte extent than the complete output axis.
Thus the pinned largest-extent splitter must split the reduction dimension.
This is a metadata fact, not a reduction-result assumption. -/
theorem torch_reduced_axis_dominates (count rows : Nat) (many : 2≤count) (nonempty : 0<rows) :
    outputExtent rows<reductionExtent count rows := by
  have lower : rows*4≤(count-1)*(rows*4) := by
    simpa only [one_mul] using Nat.mul_le_mul_right (rows*4) (show 1≤count-1 by omega)
  have strict : (rows-1)*4<rows*4 := Nat.mul_lt_mul_of_pos_right (by omega) (by decide)
  exact strict.trans_le lower

/-- A singleton reduction cannot be divided further. Subsequent output
slices retain a single reduction leaf; the four seeded accumulator program
is identical for each active output. -/
theorem torch_singleton_extent (rows : Nat) : reductionExtent 1 rows=0 := by
  simp [reductionExtent]

#axiomsClean torch_reduced_axis_dominates
#axiomsClean torch_singleton_extent
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating
