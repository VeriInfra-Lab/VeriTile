import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Snapshot
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.AtomicOrder

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.RealHost
open VeriTile.Triton Geometry Snapshot
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000

/-- Mathematical target in caller coordinates, independent of split geometry,
source statements, atomics, the reduction tree and the source evaluator. -/
def math (h : Call) (logical : Logical ℝ) (batch n : Nat) : ℝ :=
  if logical.indices batch=-1 then logical.output batch n else
    logical.output batch n+h.scaling*∑ k : Fin h.K,
      logical.input batch k.val*logical.weight (logical.indices batch).toNat n k.val

def initial (h : Call) (logical : Logical ℝ) (packed : Packed ℝ h logical 0) : BlockState where
  mem := fun r off => if r="X" then .real (packed.input off)
    else if r="W" then .real (packed.weights off)
    else if r="Y" then .real (packed.output off)
    else if r="Indices" then .of .int (packed.indices off) else .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0

def launch (h : Call) (p : Program h) (u : BlockState) : BlockState :=
  {u with regs := fun _ _ _ => none
          pids := fun ax => if ax=0 then p.split.val else if ax=1 then p.batch.val else 0
          numPids := fun ax => if ax=0 then 64 else if ax=1 then h.batches else 1}

/-- Every actual grid program succeeds under the precision-erased semantics,
including sentinel rows, with its exact contribution and full memory frame. -/
theorem every_program (h : Call) (logical : Logical ℝ) (packed : Packed ℝ h logical 0)
    (valid : ∀ b,b<h.batches → logical.indices b=-1 ∨ 0≤logical.indices b)
    (p : Program h) :
    ∃ v,exec (Kernels.source h).eraseDType.toAlgKernel (launch h p (initial h logical packed))=some v ∧
      (logical.indices p.batch.val=-1 → v.mem=(initial h logical packed).mem) ∧
      (logical.indices p.batch.val≠-1 → ∀ n : Fin h.N,
        v.readMem "Y" (outputOffset h p.batch.val n.val)=
        (initial h logical packed).readMem "Y" (outputOffset h p.batch.val n.val)+
          RealExecution.contribution h (initial h logical packed) p.batch.val
            (logical.indices p.batch.val).toNat n.val p.split.val) ∧
      (∀ r off,¬RealExecution.writes h p (logical.indices p.batch.val) r off →
        v.mem r off=(initial h logical packed).mem r off) := by
  apply RealExecution.source_executes h p (launch h p (initial h logical packed)) rfl rfl
    (logical.indices p.batch.val) (valid _ p.batch.isLt)
  simp [launch,initial,BlockState.readMemValue,BlockState.readMemTyped,packed.metadataCopy _ p.batch.isLt]

/-- Per-cell atomic execution, parameterized by the runtime's actual arbitrary
linearization. The payloads are established by every_program; event coverage
comes from AtomicOrder.fiber. No numerical commutativity is an assumption. -/
def scratchOutput (h : Call) (orders : AtomicOrder.Orders h) (logical : Logical ℝ)
    (packed : Packed ℝ h logical 0) (b : Fin h.batches) (n : Fin h.N) : ℝ :=
  if logical.indices b.val=-1 then packed.output (outputOffset h b.val n.val) else
    (orders.order b n).foldl (fun a split => a+
      RealExecution.contribution h (initial h logical packed) b.val
        (logical.indices b.val).toNat n.val split.val)
      ((initial h logical packed).readMem "Y" (outputOffset h b.val n.val))

theorem readback (h : Call) (orders : AtomicOrder.Orders h) (logical : Logical ℝ)
    (packed : Packed ℝ h logical 0)
    (valid : ∀ b,b<h.batches → logical.indices b=-1 ∨
      (0≤logical.indices b ∧ (logical.indices b).toNat<h.adapters))
    (b : Fin h.batches) (n : Fin h.N) :
    scratchOutput h orders logical packed b n=math h logical b.val n.val := by
  unfold scratchOutput math
  by_cases sentinel : logical.indices b.val=-1
  · simp only [sentinel,if_true]
    exact packed.outputCopy _ _ b.isLt n.isLt
  · simp only [sentinel,if_false]
    rw [AtomicOrder.contribution_readback]
    have selected := ((valid _ b.isLt).resolve_left sentinel).2
    have dot : (∑ k : Fin h.K,packed.input (inputOffset h b.val k.val)*
        packed.weights (weightOffset h (logical.indices b.val).toNat n.val k.val))=
        ∑ k : Fin h.K,logical.input b.val k.val*logical.weight (logical.indices b.val).toNat n.val k.val := by
      apply Finset.sum_congr rfl
      intro k _
      rw [packed.inputCopy _ _ b.isLt k.isLt,packed.weightCopy _ _ _ selected n.isLt k.isLt]
    simp [initial,BlockState.readMem,MemCell.real,MemCell.of,MemCell.readAs,
      packed.outputCopy _ _ b.isLt n.isLt,dot]


#axiomsClean every_program
#axiomsClean readback
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.RealHost
