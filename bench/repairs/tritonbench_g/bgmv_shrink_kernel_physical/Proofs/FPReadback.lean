import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPExecution
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPReadback
open VeriTile.Triton FP.Structural VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

theorem output_read (h : Call) (data : Data α) (batch split off : Nat) :
    toFloat h.outputType (((entry h batch split data).mem "Y" off).read h.outputType.toTileDType)=data.output off := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases outputType <;> simp [entry,Data.cells,Cell.read,ofFloat,toFloat]

theorem every_program (M : Algebra α) (P : Primitives α) (h : Call)
    (p : Program h) (data : Data α) (valid : data.indices p.batch.val=-1 ∨ 0 ≤ data.indices p.batch.val) :
    ∃ t,FP.Structural.exec (machine M P) (Surface.source h) (entry h p.batch.val p.split.val data)=some t ∧
      (data.indices p.batch.val=-1 → t.mem=(entry h p.batch.val p.split.val data).mem) ∧
      (data.indices p.batch.val ≠ -1 → ∀ n : Fin h.N,
        t.mem "Y" (outputOffset h p.batch.val n.val)=
          .mk h.outputType.toTileDType (ofFloat _ (P.atomic h.outputType
            (data.output (outputOffset h p.batch.val n.val))
            (payload M P h data p.batch.val (data.indices p.batch.val).toNat p.split.val
              ⟨n.val,n.isLt.trans_le h.rowsCovered⟩)))) ∧
      (∀ (r : RegionName) (off : Nat),¬(data.indices p.batch.val ≠ -1 ∧ r="Y" ∧ ∃ n : Fin h.N,
        off=outputOffset h p.batch.val n.val) → t.mem r off=(entry h p.batch.val p.split.val data).mem r off) := by
  obtain ⟨t,exec,memory⟩ := FPExecution.source_executes M P h p.batch.val p.split.val data valid
  refine ⟨t,exec,?_,?_,?_⟩
  · intro sentinel
    simpa only [sentinel,if_true] using memory
  · intro active n
    rw [memory,if_neg active,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.tile_value]
    rw [output_read]
  · intro r off outside
    rw [memory]
    by_cases sentinel : data.indices p.batch.val=-1
    · simp only [sentinel,if_true]
    · rw [if_neg sentinel]
      apply VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.tile_frame
      exact fun hit => outside ⟨sentinel,hit⟩

/-- Payload independence from output and metadata allows arbitrary atomic
interleavings after logical snapshots. This is structural, with no FP laws. -/
theorem payload_independent (M : Algebra α) (P : Primitives α) (h : Call)
    (a b : Data α) (input : a.input=b.input) (weights : a.weights=b.weights) (scaling : a.scaling=b.scaling)
    (batch slot split : Nat) : payload M P h a batch slot split=payload M P h b batch slot split := by
  have acc : ∀ c,accumulator M P h a batch slot split c=accumulator M P h b batch slot split c := by
    intro c
    induction c with
    | zero => rfl
    | succ c ih =>
      funext n
      have prod : VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.products M h a batch slot split c=VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.products M h b batch slot split c := by
        funext i
        simp [VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.products,loaded,weight,input,weights]
      simp only [accumulator,ih,sums,prod]
  funext n
  simp only [payload,acc,scaling]

#axiomsClean every_program
#axiomsClean payload_independent
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPReadback
