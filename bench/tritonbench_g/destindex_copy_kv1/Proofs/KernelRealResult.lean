import bench.tritonbench_g.destindex_copy_kv1.Proofs.KernelInterface
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.DestindexCopyKv1.KernelRealResult
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete Real source launch on raw typed memory and logical arrays. -/
theorem destindex_copy_kv1_kernel_real (rest : Algebra ℝ) (c : Config) (ps : List Nat) (s : State ℝ)
    (source : Nat → ℝ) (dest : Nat → Nat) (reads : Grid.Bound s source dest)
    (unique : ∀ p q,p<c.count → q<c.count → dest p=dest q → p=q)
    (order : ∀ p,p∈ps ↔ p<c.count) :
    ∃ final, KernelInterface.Run rest c ps s final ∧
      RowCopy c dest source (fun x => (s.mem outputRegion x).read .real)
        (fun x => (final.mem outputRegion x).read .real) ∧
      (∀ r o,(r≠outputRegion ∨ ∀ p,p<c.count → ∀ head dim,
        head<c.heads → dim<c.dims → o≠flat c (dest p) head dim) →
        final.mem r o=s.mem r o) :=
  KernelInterface.complete rest c ps s source dest reads unique order

#axiomsClean destindex_copy_kv1_kernel_real
#specNonCircular RowCopy avoiding [kernel,Grid.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv1.KernelRealResult
