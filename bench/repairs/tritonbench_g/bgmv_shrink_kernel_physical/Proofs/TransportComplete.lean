import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TransportTail

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.MemoryTransport
open VeriTile.Triton FP.Structural FPProgram FPValues FPSetup FPBody
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

/-- The complete original source on arbitrary backing memory. Only its exact
active metadata/input/weight cells are fixed; its atomic reads use the actual
current output memory. Sentinel programs preserve every original raw cell. -/
theorem source_transport (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split : Nat) (data : Data α) (memory : RegionName → Nat → Cell α)
    (valid : data.indices batch = -1 ∨ 0 ≤ data.indices batch)
    (index : memory "Indices" batch = .mk .int (data.indices batch))
    (inputRead : data.indices batch ≠ -1 → ∀ k, k < h.K →
      memory "X" (inputOffset h batch k) =
        .mk (dtype h.storage).toTileDType (ofFloat _ (data.input (inputOffset h batch k))))
    (weightRead : data.indices batch ≠ -1 → ∀ n k, n < h.N → k < h.K →
      memory "W" (weightOffset h (data.indices batch).toNat n k) =
        .mk (dtype h.storage).toTileDType
          (ofFloat _ (data.weights (weightOffset h (data.indices batch).toNat n k)))) :
    ∃ t, FP.Structural.exec (machine M P) (Surface.source h)
        (rebase (entry h batch split data) memory) = some t ∧
      t.mem = (if data.indices batch = -1 then memory else
        (VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.tile h P batch
          (payload M P h data batch (data.indices batch).toNat split)
          (rebase (entry h batch split data) memory)).mem) := by
  have condition : FP.Structural.evalOp (machine M P) none FPProgram.guard
      (rebase (preguard M P h batch split data).val memory) =
      some (fun _ => decide (data.indices batch ≠ -1)) :=
    FPExecution.condition M P h batch split data
  rw [FPExecution.exec_code, complete_code, run_append,
    prefix_transport M P h batch split data memory index, Option.bind_some]
  by_cases sentinel : data.indices batch = -1
  · refine ⟨rebase (preguard M P h batch split data).val memory, ?_, ?_⟩
    · apply FPExecution.Control.single
      apply FPExecution.Control.skip
      simpa only [sentinel, ne_self_iff_false, decide_false] using condition
    · simp only [sentinel, if_true, rebase]
  · have nonnegative := valid.resolve_left sentinel
    have metadata : data.indices batch = ((data.indices batch).toNat:Int) :=
      (Int.toNat_of_nonneg nonnegative).symm
    obtain ⟨after, _, ready, actualLoop⟩ := loop_transport M P h batch split
      (data.indices batch).toNat data metadata memory (inputRead sentinel) (weightRead sentinel)
    let final := finish_rebased M P h batch split (data.indices batch).toNat data after ready memory
    refine ⟨final.val, ?_, ?_⟩
    · apply FPExecution.Control.single
      apply FPExecution.Control.pass
      · simpa [sentinel] using condition
      · rw [complete_active, run_append,
          setup M P h batch split (data.indices batch).toNat data metadata memory,
          Option.bind_some]
        simp only [run, actualLoop, Option.bind_some]
        exact final.property
    · simp only [sentinel, if_false]
      rw [finish_rebased_memory]
      unfold VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.tile
      apply VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.fold_memory
      · rfl
      · intro i u v same
        by_cases active : i.1.val < h.N <;> simp only [active, if_true, if_false]
        · simp only [State.write, same]
        · exact same

/-- Exact source result, selected-cell atomic update and raw exterior frame.
No output value or tag is constrained merely to establish source execution. -/
theorem every_program_transport (M : Algebra α) (P : Primitives α) (h : Call)
    (p : Program h) (data : Data α) (memory : RegionName → Nat → Cell α)
    (valid : data.indices p.batch.val = -1 ∨ 0 ≤ data.indices p.batch.val)
    (index : memory "Indices" p.batch.val = .mk .int (data.indices p.batch.val))
    (inputRead : data.indices p.batch.val ≠ -1 → ∀ k, k < h.K →
      memory "X" (inputOffset h p.batch.val k) =
        .mk (dtype h.storage).toTileDType (ofFloat _ (data.input (inputOffset h p.batch.val k))))
    (weightRead : data.indices p.batch.val ≠ -1 → ∀ n k, n < h.N → k < h.K →
      memory "W" (weightOffset h (data.indices p.batch.val).toNat n k) =
        .mk (dtype h.storage).toTileDType
          (ofFloat _ (data.weights (weightOffset h (data.indices p.batch.val).toNat n k)))) :
    ∃ t, FP.Structural.exec (machine M P) (Surface.source h)
        (rebase (entry h p.batch.val p.split.val data) memory) = some t ∧
      (data.indices p.batch.val = -1 → t.mem = memory) ∧
      (data.indices p.batch.val ≠ -1 → ∀ n : Fin h.N,
        t.mem "Y" (outputOffset h p.batch.val n.val) =
          .mk h.outputType.toTileDType (ofFloat _ (P.atomic h.outputType
            (toFloat h.outputType ((memory "Y" (outputOffset h p.batch.val n.val)).read h.outputType.toTileDType))
            (payload M P h data p.batch.val (data.indices p.batch.val).toNat p.split.val
              ⟨n.val, n.isLt.trans_le h.rowsCovered⟩)))) ∧
      (∀ (r : RegionName) (off : Nat),
        ¬ (data.indices p.batch.val ≠ -1 ∧ r = "Y" ∧ ∃ n : Fin h.N,
          off = outputOffset h p.batch.val n.val) → t.mem r off = memory r off) := by
  obtain ⟨t, executed, result⟩ := source_transport M P h p.batch.val p.split.val data memory
    valid index inputRead weightRead
  refine ⟨t, executed, ?_, ?_, ?_⟩
  · intro sentinel
    simpa only [sentinel, if_true] using result
  · intro active n
    rw [result, if_neg active,
      VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.tile_value]
    rfl
  · intro r off outside
    rw [result]
    by_cases sentinel : data.indices p.batch.val = -1
    · simp only [sentinel, if_true]
    · rw [if_neg sentinel]
      exact VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.tile_frame
        h P p.batch.val _ _ r off (fun hit => outside ⟨sentinel, hit⟩)

#axiomsClean source_transport
#axiomsClean every_program_transport
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.MemoryTransport
