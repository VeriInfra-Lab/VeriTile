import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.IOGrid
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.RealArrayReference

namespace VeriTile.Bench.Repairs.MixedSparseAttention.RealIO
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked
noncomputable section
set_option maxHeartbeats 2000000

theorem implements (c : Config) (sep : Separated c) (layout : Contiguous c)
    (hbm : 0<c.BM) (hbn : 0<c.BN) (work : List Program)
    (order : work.Perm (programs (grid c))) :
    (io c work).ImplementsOn (domain c) (mathematical c) := by
  intro xs hd s hin
  have h:=binding c work xs s hin
  obtain ⟨safe,final,ran,result,frame⟩:=KernelIO.whole Real.algebra Real.dotModel c
    sep layout hbm hbn work order (lifted c xs) hd s h
  have safe' : ∀ bounds,(io c work).InBounds bounds xs →
      ScheduleSafe bounds ((io c work).schedule xs) s := by
    intro bounds hb
    apply safe bounds
    refine ⟨hb.1,hb.2.1,?_,by intro k;exact Fin.elim0 k⟩
    intro o i active
    change Fin 1 at o
    fin_cases o
    exact hb.2.2.1 (0 : Fin 1) i active
  have frame' : (io c work).Frame xs s final := by
    intro r off outside
    apply frame r off
    refine ⟨?_,by intro k;exact Fin.elim0 k⟩
    intro o i active
    change Fin 1 at o
    fin_cases o
    exact outside.1 (0 : Fin 1) i active
  refine ⟨safe',final,ran,?_,frame'⟩
  intro ⟨o,ho⟩ i active
  have hz : o=0 := by change o<1 at ho;omega
  subst o
  let q:=KernelIO.coordinate c i
  have live : KernelGrid.live c s q := by
    simpa [KernelGrid.live,q,KernelIO.coordinate,Raw.hz,Raw.row,
      KernelIO.active,h.length] using active
  have output:=result.output q live
  have formula:=Real.source_value_real c layout s (realValues s) (real_reads c xs s h) hbm hbn
    i.1.val i.2.1.val i.2.2.1.val i.2.2.1.isLt
  have active' : i.2.1.val<(s.mem (Region.cast c.seq) (i.1.val/c.H)).read .nat := live
  simp only [if_pos active',Real.logicalExpected] at formula
  have exactValue : final.mem c.out (KernelIO.address c i)=
      Cell.mk .real (some (mathematical c xs (0 : Fin 1) i)) := by
    change final.mem c.out (Raw.offset q)=_
    rw [output]
    simpa only [q,KernelIO.coordinate,KernelGrid.owner,KernelGrid.rowLane,Raw.hz,Raw.row,Raw.dim,
      ofFloat,mathematical_source c xs s h i] using
      congrArg (fun v : Real.R=>Cell.mk .real v) formula
  change mapCell (fun x : Real.R=>x.unbotD 0) (final.mem c.out (KernelIO.address c i))=_
  rw [exactValue]
  rfl

#axiomsClean implements
end
end VeriTile.Bench.Repairs.MixedSparseAttention.RealIO
