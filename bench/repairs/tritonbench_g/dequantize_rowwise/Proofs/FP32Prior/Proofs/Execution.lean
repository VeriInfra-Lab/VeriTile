import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Kernels
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Execution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def scalar (M : Algebra α) (scale x : α) : α :=
  M.cast none .real .fp16 (M.binary (some .fp32) .real .mul
    (M.binary (some .fp32) .real .mul (M.fp32Load scale) (M.fp32Load x))
    (M.literal (some .fp32) .real (1/127)))

def offset (p : Parameters) (s : State α) (i : TileIndex [p.width]) : Nat :=
  s.pids 0*p.cols+i.1.val

def active (p : Parameters) (i : TileIndex [p.width]) : Prop := i.1.val<p.cols
instance (p : Parameters) (i : TileIndex [p.width]) : Decidable (active p i) := inferInstanceAs (Decidable (i.1.val<p.cols))

def cell (M : Algebra α) (p : Parameters) (s : State α) (o : Nat) : Cell α :=
  .mk .fp16 (scalar M ((s.mem scaleRegion (s.pids 0)).read .real) ((s.mem xRegion o).read .real))

def expected (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl (fun t i =>
    if active p i then t.write outRegion (offset p s i) (cell M p s (offset p s i)) else t) s

/-- Constructed execution of all nine source statements, from arbitrary
registers, payloads and scalar primitives. No kernel-result assumption. -/
def execution (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (kernel p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [kernel,source,FP.Structural.exec,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    natLt,numericLt,bop,store,Region.cast,ComputeDType.eraseDType]
  rfl

theorem execution_memory (M : Algebra α) (p : Parameters) (s : State α) :
    (execution M p s).val.mem=(expected M p s).mem := by
  simp [execution,expected,offset,active,cell,scalar,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,masked_scatter_memory,State.setReg]
  congr 1
  funext memory i
  by_cases bound : i.1.val<p.cols
  · simp [bound,numeric,ofFloat,toFloat]
  · simp [bound]

theorem expected_readback (M : Algebra α) (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) (hi : active p i) :
    (expected M p s).mem outRegion (offset p s i)=cell M p s (offset p s i) := by
  apply masked_readback_active _ _ _ _ _ i hi
  intro a b _ _ equality
  apply Prod.ext _ (Subsingleton.elim _ _)
  apply Fin.ext
  exact Nat.add_left_cancel equality

theorem expected_frame (M : Algebra α) (p : Parameters) (s : State α)
    (r : RegionName) (o : Nat)
    (miss : r≠outRegion ∨ ∀ i,active p i → o≠offset p s i) :
    (expected M p s).mem r o=s.mem r o := by
  exact State.masked_scatter_frame _ _ _ _ _ r o (miss.imp id (fun h i _ hi=>h i hi)) s

theorem complete_correct (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t,FP.Structural.exec M (kernel p) s=some t ∧
      (∀ d,d<p.cols → t.mem outRegion (s.pids 0*p.cols+d)=
        cell M p s (s.pids 0*p.cols+d)) ∧
      (∀ r o,(r≠outRegion ∨ ∀ d,d<p.cols → o≠s.pids 0*p.cols+d) → t.mem r o=s.mem r o) := by
  refine ⟨(execution M p s).val,(execution M p s).property,?_,?_⟩
  · intro d hd
    rw [execution_memory]
    exact expected_readback M p s (⟨d,lt_of_lt_of_le hd p.covers⟩,PUnit.unit) hd
  · intro r o miss
    rw [execution_memory]
    exact expected_frame M p s r o (miss.imp id (fun h i hi=>h i.1.val hi))

#axiomsClean execution_memory
#axiomsClean complete_correct
end
end VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Execution
