import bench.tritonbench_g.matrix_reduction.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.MatrixReduction
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTileKernelIO₁

/-- Per-kernel output summary for `load_reduce_kernel`: the DSL surface lowers to
the algorithm layer, and the row-vector store to `y_ptr` is compute-correct —
every row lane `i` holds the row-wise maximum `matrixReduceSpec`. -/
specification load_reduce_kernel_output_summary
    (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat)
    (s : BlockState) :
    (∃ alg, (load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
        BLOCK_M BLOCK_N).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
        BLOCK_M BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (y_ptr, i.val))
      (expected := fun i =>
        matrixReduceSpec s x_ptr stride_xm stride_xn BLOCK_M BLOCK_N i) := by
  refine ⟨⟨_, mathematical_supported ..⟩, ?_⟩
  exact load_reduce_kernel_compute_correct x_ptr y_ptr stride_xm stride_xn
    stride_y BLOCK_M BLOCK_N s

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `matrix_reduction.py`'s
`load_reduce_kernel`: for every disjoint flat placement of the two buffers, every
launch state whose `BLOCK_M × BLOCK_N` input tile holds `xs`, the translated
pointer kernel terminates, `y_ptr[i]` holds the genuine row-wise maximum of row
`i` of `xs`, and every other memory cell is unchanged.

Dimension-general in both strides, `BLOCK_M` and `BLOCK_N`. Honest
side-condition: `0 < BLOCK_N` — with an empty reduction axis `tl.max` faults, so
termination genuinely fails there; the same hypothesis witnesses a write-active
lane, turning the skin's lane-wise frame disjunct into the row-vector one. -/
specification matrix_reduction_io_correctness (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N) :
    matrixReduceIO x_ptr y_ptr stride_xm stride_xn stride_y BLOCK_M BLOCK_N
      ⊨ fun _pid xs idx => matrixReduceSpecOf BLOCK_M BLOCK_N xs idx := by
  refine MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact matrix_reduction_flattenOk x_ptr y_ptr stride_xm stride_xn stride_y
      BLOCK_M BLOCK_N
  · intro bounds s h1 h2
    exact matrix_reduction_traceSafe x_ptr y_ptr stride_xm stride_xn stride_y
      BLOCK_M BLOCK_N hBN bounds s (fun idx => h1 idx trivial)
      (fun i => h2 (i, ⟨0, hBN⟩, PUnit.unit) rfl)
  · intro s₀ xs hin
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      matrix_reduction_region_run x_ptr y_ptr stride_xm stride_xn stride_y
        BLOCK_M BLOCK_N hBN s₀ xs (fun idx => hin idx trivial)
    refine ⟨s1, hexec, fun idx _ => hval idx, ?_⟩
    intro r o hcond
    refine hframe r o ?_
    rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i => h (i, ⟨0, hBN⟩, PUnit.unit) rfl



end VeriTile.Bench.TritonBenchG.MatrixReduction
