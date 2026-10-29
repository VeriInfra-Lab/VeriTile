import bench.repairs.tritonbench_g.kldiv_ops.Proofs.IOInterface
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Native

namespace VeriTile.Bench.Repairs.KldivOps.NativeIO
open VeriTile Triton FP.Structural Interface
open KldivTriton.Native (Word Node atom)
noncomputable section

def forwardFormula (ports : Contiguous.Native.Ports) (plans : Contiguous.Native.Plans)
    (p : Parameters) (flag : Bool) (red : Nat) (eps : BitVec 64)
    (values : ForwardValues Word p) (i : Fin (outputLength p red)) : Word :=
  if red=0 then Contiguous.Native.loss ports flag (ports.bindEpsilon eps)
    (predictionAt values i.val) (targetAt values i.val)
  else Contiguous.Native.row ports plans p.cols flag (ports.bindEpsilon eps)
    (fun col => predictionAt values (i.val*p.cols+col))
    (fun col => targetAt values (i.val*p.cols+col))

def backwardFormula (ports : Contiguous.Native.Ports) (p : Parameters) (flag : Bool)
    (values : BackwardValues Word p) (i : Fin (p.rows*p.cols)) : Word :=
  Contiguous.Native.derivative ports flag (backwardTargetAt values i.val)

theorem forward (ports : Contiguous.Native.Ports) (plans : Contiguous.Native.Plans)
    (p : Parameters) (flag : Bool) (red : Nat) (eps : BitVec 64)
    (block : p.block=Host.blockSize p.cols) (hy : p.Y≠p.O) (ht : p.T≠p.O) :
    (forwardIO (Contiguous.Native.algebra ports plans) p flag red (atom (ports.bindEpsilon eps))
      atom Node.value).Implements (fun values => result (forwardFormula ports plans p flag red eps values)) := by
  have positive : 0<p.block := block ▸ Host.block_positive p.cols
  apply forward_implements _ p flag red _ positive atom Node.value
  intro values initial predictionRead targetRead
  obtain ⟨final,ran,outputs,frame⟩ := KernelResults.forward (Contiguous.Native.algebra ports plans)
    p flag red (atom (ports.bindEpsilon eps)) initial positive hy ht
  refine ⟨final,ran,?_,frame⟩
  intro i
  rw [outputs i.val i.isLt]
  simp only [mapCell,mapValue]
  congr 1
  by_cases mode : red=0
  · have hi : i.val<p.rows*p.cols := by simpa only [outputLength,if_pos mode] using i.isLt
    simp only [forwardFormula,if_pos mode,predictionRead _ hi,targetRead _ hi,Contiguous.Native.scalar_lowering]
    rfl
  · have hr : i.val<p.rows := by simpa only [outputLength,if_neg mode] using i.isLt
    simp only [forwardFormula,if_neg mode]
    have reads := FPGraph.accumulate_congr (Contiguous.Native.algebra ports plans) p.cols p.block flag
      (atom (ports.bindEpsilon eps))
      (fun col => (initial.mem p.Y (i.val*p.cols+col)).read .real)
      (fun col => (initial.mem p.T (i.val*p.cols+col)).read .real)
      (fun col => atom (predictionAt values (i.val*p.cols+col)))
      (fun col => atom (targetAt values (i.val*p.cols+col))) 0
      (Tile.zero (Contiguous.Native.algebra ports plans))
      (fun col hc => predictionRead _ (KernelResults.offset_lt p i.val col hr hc))
      (fun col hc => targetRead _ (KernelResults.offset_lt p i.val col hr hc))
    unfold FPGraph.row
    rw [reads,block]
    exact Contiguous.Native.row_lowering ports plans p.cols flag (ports.bindEpsilon eps) _ _

theorem backward (ports : Contiguous.Native.Ports) (plans : Contiguous.Native.Plans)
    (p : Parameters) (flag : Bool) (positive : 0<p.block) (ht : p.T≠p.O) :
    (backwardIO (Contiguous.Native.algebra ports plans) p flag atom Node.value).Implements
      (fun values => result (backwardFormula ports p flag values)) := by
  apply backward_implements _ p flag positive atom Node.value
  intro values initial targetRead
  obtain ⟨final,ran,outputs,frame⟩ := KernelResults.backward (Contiguous.Native.algebra ports plans)
    p flag initial positive ht
  refine ⟨final,ran,?_,frame⟩
  intro i
  rw [outputs i.val i.isLt,targetRead _ i.isLt,Contiguous.Native.backward_lowering]
  rfl
end
end VeriTile.Bench.Repairs.KldivOps.NativeIO
