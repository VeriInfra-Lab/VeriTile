import bench.tritonbench_g.cross_entropy1.Proofs.TensorOps

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton FP.Structural Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000

theorem columnValues_eq {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {s : World α σ} {t : Tensor} {blocks rows : Nat} {v : Nat → α}
    (array : FloatArray obs s t [blocks,rows] v) (row : Nat) (hr : row<rows) :
    columnValues obs s t blocks rows row=(List.range blocks).map (fun block => v (block*rows+row)) := by
  apply List.map_congr_left
  intro block hb
  apply array.values
  have bound : block<blocks := List.mem_range.mp hb
  have product := Nat.mul_le_mul_right rows (Nat.succ_le_of_lt bound)
  simp only [Nat.succ_mul] at product
  simp only [Tensor.numel,array.shapeEq,List.prod_cons,List.prod_nil,Nat.mul_one]
  omega

def localResult {α : Type} (reduce : List α → α) (count rows : Nat) (v : Nat → α) (row : Nat) : α :=
  if count>1 then reduce ((List.range count).map (fun block => v (block*rows+row))) else v row

structure LocalValues {α σ : Type} [Inhabited α] (obs : Observations α σ) (A : TensorArithmetic α)
    (count rows : Nat) (loss lse : Nat → α) (initial : World α σ) (pair : Tensor × Tensor) (final : World α σ) : Prop where
  losses : FloatArray obs final pair.1 [rows] (localResult A.sum count rows loss)
  lse : FloatArray obs final pair.2 [rows] (localResult A.logsumexp count rows lse)
  separate : pair.1.region≠pair.2.region
  frame : Preserves obs initial final

/-- The actual local reduction consumes the source stores in flattened
block-major order. Its result follows only from the two local library calls. -/
theorem local_reduction_values {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (ops : TensorLibrary obs A rt)
    (count rows : Nat) (losses lse : Tensor) (initial : World α σ) (pair : Tensor × Tensor) (final : World α σ)
    (lossValues lseValues : Nat → α)
    (lossInput : FloatArray obs initial losses (if count>1 then [count,rows] else [rows]) lossValues)
    (lseInput : FloatArray obs initial lse (if count>1 then [count,rows] else [rows]) lseValues)
    (separate : losses.region≠lse.region)
    (trace : LocalTrace rt count losses lse initial pair final) :
    LocalValues obs A count rows lossValues lseValues initial pair final := by
  cases trace with
  | direct one =>
    refine ⟨?_,?_,separate,Preserves.refl obs initial⟩
    · have eq : localResult A.sum count rows lossValues=lossValues := by funext row; simp [localResult,one]
      simpa only [if_neg one,eq] using lossInput
    · have eq : localResult A.logsumexp count rows lseValues=lseValues := by funext row; simp [localResult,one]
      simpa only [if_neg one,eq] using lseInput
  | reduced many localLSE localLoss logged summed lseCall sumCall =>
    have li : FloatArray obs initial losses [count,rows] lossValues := by
      simpa only [if_pos many] using lossInput
    have si : FloatArray obs initial lse [count,rows] lseValues := by
      simpa only [if_pos many] using lseInput
    have first := ops.logsumexp lse count rows initial localLSE logged si.live si.shapeEq si.dtype lseCall
    have preserved := li.preserve first.allocation.frame
    have second := ops.sum losses count rows logged localLoss _ preserved.live preserved.shapeEq preserved.dtype sumCall
    refine ⟨?_,?_,fresh_ne_live obs _ _ _ _ _ _ second.allocation _ first.allocation.live,
      first.allocation.frame.trans second.allocation.frame⟩
    · refine ⟨second.allocation.shapeEq,second.allocation.dtypeEq,second.allocation.live,?_⟩
      intro row hr
      have bound : row<rows := by simpa [Tensor.numel,second.allocation.shapeEq] using hr
      rw [second.values row hr,columnValues_eq preserved row bound]
      simp only [localResult,if_pos many]
    · have saved := first.array.preserve second.allocation.frame
      refine ⟨saved.shapeEq,saved.dtype,saved.live,?_⟩
      intro row hr
      have bound : row<rows := by simpa [Tensor.numel,saved.shapeEq] using hr
      rw [saved.values row hr,columnValues_eq si row bound]
      simp only [localResult,if_pos many]

#axiomsClean columnValues_eq
#axiomsClean local_reduction_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
