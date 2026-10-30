import bench.repairs.tritonbench_g.matrix_transpose.IO
import bench.repairs.tritonbench_g.matrix_transpose.Proofs.Indices
import bench.tritonbench_g.matrix_transpose.Proofs.Real
import bench.tritonbench_g.matrix_transpose.Proofs.Source

/-! Kernel-only mathematical proof. The input is initial logical memory;
allocation, random sampling, launch orchestration and returned identity are
outside this proof. The flat-memory IO theorem supplies disjoint placement. -/
namespace VeriTile.Bench.Repairs.MatrixTranspose.KernelReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.MatrixTranspose
open scoped VeriTile.Triton.Masked2DKernelIO₁
noncomputable section

theorem source_projection (m d : Nat) :
    (source "matrix" "out" m d).eraseDType.toAlgKernel.body =
      (kernel "matrix" "out" (max d 1) 1 (max m 1) 1 m d).toAlgKernel.body := by
  change (VeriTile.Bench.TritonBenchG.MatrixTranspose.typedKernel Bool.false
    "matrix" "out" (max d 1) 1 (max m 1) 1 m d).eraseDType.toAlgKernel.body = _
  rw [typed_projection]
  rfl

/-- The contiguous output has a distinct address for every output coordinate. -/
theorem output_injective (m d : Nat) :
    Function.Injective (fun i : TileIndex [d,m] => outAddr (max m 1) 1 i) := by
  intro a b eq
  have ha : a.2.1.val < m := a.2.1.isLt
  have hb : b.2.1.val < m := b.2.1.isLt
  have hm : 1 ≤ m := by omega
  simp only [outAddr, Nat.max_eq_left hm, Nat.mul_one] at eq
  have lower := congrArg (fun o => o % m) eq
  dsimp only at lower
  rw [Nat.mul_add_mod_of_lt ha, Nat.mul_add_mod_of_lt hb] at lower
  have upper : a.1.val = b.1.val :=
    Nat.mul_right_cancel (by omega : 0 < m) (by omega)
  apply Prod.ext (Fin.ext upper)
  exact Prod.ext (Fin.ext lower) (Subsingleton.elim _ _)

theorem implements (m d : Nat) :
    transposeIO m d ⊨ fun _ _ xs j => xs j := by
  refine Masked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · change ((source "matrix" "out" m d).eraseDType.toAlgKernel).FlattenOk
    simpa only [Kernel.FlattenOk, source_projection] using
      kernel_flattenOk "matrix" "out" (max d 1) 1 (max m 1) 1 m d
  · intro bounds s hin hout _
    change Kernel.TraceSafe bounds ((source "matrix" "out" m d).eraseDType.toAlgKernel) s
    simpa only [Kernel.TraceSafe, source_projection] using
      kernel_traceSafe "matrix" "out" (max d 1) 1 (max m 1) 1 m d bounds s
        (fun j => hin j trivial) (fun j => hout j trivial)
  · intro s xs hx
    obtain ⟨t, he, hv, hf⟩ :=
      kernel_region_run "matrix" "out" (max d 1) 1 (max m 1) 1 m d s xs
        (output_injective m d) (fun j => hx j trivial)
    refine ⟨t, ?_, fun j _ => hv j, fun r o h _ => hf r o ?_⟩
    · change exec ((source "matrix" "out" m d).eraseDType.toAlgKernel) s = some t
      simpa only [exec, source_projection] using he
    · exact h.imp_right (fun hh j => hh j trivial)

/-- Independent coordinate formula plus the complete exterior frame, from
arbitrary initial memory. No execution-success premise is assumed. -/
theorem coordinates (m d : Nat) (h : InputContract m d) (s : BlockState) :
    ∃ t, exec ((source "matrix" "out" m d).eraseDType.toAlgKernel) s = some t ∧
      (∀ row : Fin m, ∀ col : Fin d,
        t.readMem "out" (col.val*m+row.val) = formula (s.readMem "matrix") d row.val col.val) ∧
      (∀ (r : RegionName) o, r ≠ "out" ∨ m*d ≤ o → t.mem r o = s.mem r o) := by
  let xs : Fin (d*m) → ℝ := fun j =>
    s.readMem "matrix" (matrixAddr (max d 1) 1 (Lane2D.decode j))
  obtain ⟨t, he, hv, hf⟩ := kernel_region_run "matrix" "out" (max d 1) 1
    (max m 1) 1 m d s xs (output_injective m d) (fun _ => rfl)
  refine ⟨t, ?_, ?_, ?_⟩
  · simpa only [exec, source_projection] using he
  · intro row col
    have value := hv (Lane2D.encode (col,row,()))
    simpa [xs, outAddr, matrixAddr, Nat.max_eq_left h.positive.1,
      Nat.max_eq_left h.positive.2, formula] using value
  · intro r o outside
    apply hf r o
    rcases outside with hn | ho
    · exact Or.inl hn
    · right
      intro j eq
      have bound : outAddr (max m 1) 1 (Lane2D.decode j) < m*d := by
        simpa [outAddr, Nat.max_eq_left h.positive.1, Nat.mul_comm] using
          Indices.flat_bound d m (Lane2D.decode j).1.val (Lane2D.decode j).2.1.val
            (Lane2D.decode j).1.isLt (Lane2D.decode j).2.1.isLt
      omega

#axiomsClean source_projection
#axiomsClean implements
#axiomsClean coordinates
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixTranspose.KernelReal
