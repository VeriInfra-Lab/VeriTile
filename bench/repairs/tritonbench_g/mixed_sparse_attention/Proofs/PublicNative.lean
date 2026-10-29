import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.IOGrid
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.NativeArrayReference

namespace VeriTile.Bench.Repairs.MixedSparseAttention.NativeIO
open VeriTile Triton FP.Structural
open Whole002 Source Whole Native Checked
noncomputable section
set_option maxHeartbeats 2000000

theorem observes (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config)
    (sep : Separated c) (layout : Contiguous c) (hbm : 0<c.BM) (hbn : 0<c.BN)
    (work : List Program) (order : work.Perm (programs (grid c))) :
    (io rt I c work).ObservesOn (observe c I) (KernelIO.domain c) (formula rt I c hbm) := by
  intro xs hd s hin
  have h:=KernelIO.input_reads (algebra rt) (dotModel c.storage I) c work xs s hin
  obtain ⟨safe,final,ran,result,frame⟩:=KernelIO.whole (algebra rt) (dotModel c.storage I)
    c sep layout hbm hbn work order xs hd s h
  refine ⟨safe,final,ran,?_,frame⟩
  intro ⟨o,ho⟩ i active
  have hz : o=0 := by change o<1 at ho;omega
  subst o
  let q:=KernelIO.coordinate c i
  have live : KernelGrid.live c s q := by
    simpa [KernelGrid.live,q,KernelIO.coordinate,Raw.hz,Raw.row,
      KernelIO.active,h.length] using active
  have output:=result.output q live
  change mapCell (observe c I (0 : Fin 1) i)
    (mapCell id (final.mem c.out (Raw.offset q)))=_
  rw [output]
  simp only [mapCell,mapValue,ofFloat,id_eq,
    expression_source rt I c xs s h (KernelGrid.owner c q) (KernelGrid.rowLane c hbm q)]
  rfl

#axiomsClean observes
end
end VeriTile.Bench.Repairs.MixedSparseAttention.NativeIO
