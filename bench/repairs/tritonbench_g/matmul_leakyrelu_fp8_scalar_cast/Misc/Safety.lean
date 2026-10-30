import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Contract
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked.Execution

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast
open VeriTile.Triton
open MatmulLeakyrelu.KernelOnly MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
noncomputable section

/-- Every raw lane coordinate and clamped source pointer uses representable
signed64 element arithmetic. All premises are actual Tensor metadata/storage
facts plus legal grid/lane indices; there is no added Python shape restriction. -/
theorem formed_source_addresses (r : Request) (c : OutputTensor) (h : Contract r c)
    (hm : 0 < r.A.rows) (hn : 0 < r.B.columns) (hk : 0 < r.A.columns)
    (tile iteration rm cn kl : Nat) (ht : tile < (arguments r).problem.tiles)
    (hi : iteration < (arguments r).problem.kTiles) (hr : rm < 128) (hc : cn < 256) (hl : kl < 64) :
    let p := (arguments r).problem
    p.row tile rm < 2^63 ∧ p.column tile cn < 2^63 ∧ p.k iteration kl < 2^63 ∧
    ((arguments r).A.address (p.row tile rm%r.A.rows) (clamp r.A.columns (p.k iteration kl))).2 < 2^63 ∧
    ((arguments r).B.address (clamp r.A.columns (p.k iteration kl)) (p.column tile cn%r.B.columns)).2 < 2^63 ∧
    ((outputDescriptor r c).address (clamp r.A.rows (p.row tile rm)) (clamp r.B.columns (p.column tile cn))).2 < 2^63 :=
  formed_addresses (arguments r).problem ⟨(arguments r).A,(arguments r).B,outputDescriptor r c⟩
    (layout r c h) hm hn hk tile iteration rm cn kl ht hi hr hc hl

/-- Exact signed64 meaning of the nine repaired scalar casts follows from
ordinary descriptor bounds. Cast spelling carries no numerical hypothesis. -/
theorem scalar_casts_exact (r : Request) (c : OutputTensor) (h : Contract r c) :
    (∀ x ∈ [r.A.rows,r.B.columns,r.A.columns,r.A.strideRow,r.A.strideColumn,
      r.B.strideRow,r.B.strideColumn,max r.B.columns 1,1],
      (BitVec.ofNat 64 x).toInt = (x : Int)) := by
  intro x hx
  simp only [List.mem_cons,List.mem_singleton] at hx
  rcases hx with hh | hh | hh | hh | hh | hh | hh | hh | hh
  · subst x; exact signed64_exact _ h.aABI.rows
  · subst x; exact signed64_exact _ h.bABI.columns
  · subst x; exact signed64_exact _ h.aABI.columns
  · subst x; exact signed64_exact _ h.aABI.rowStride
  · subst x; exact signed64_exact _ h.aABI.columnStride
  · subst x; exact signed64_exact _ h.bABI.rowStride
  · subst x; exact signed64_exact _ h.bABI.columnStride
  · subst x; exact signed64_exact _ h.outputABI.rowStride
  · rcases hh with hh | hh
    · subst x; exact signed64_exact _ (by decide)
    · simp at hh

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast
