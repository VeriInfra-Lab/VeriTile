import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.Cleanup

namespace VeriTile.Bench.Repairs.BmmChunkFwd.Cleanup
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section

/-- Every cleared cell has one program and one lane. This excludes duplicate
writers even though all stores happen to use the same zero word. -/
theorem writer_unique (h : Host.Call) (p q : PID h) (i i' : Fin 32) (j j' : Fin 64)
    (hp : active h p i j) (hq : active h q i' j')
    (eq : offset h p i j=offset h q i' j') : p=q ∧ i=i' ∧ j=j' := by
  obtain ⟨base,col⟩ := Host.mul_add_inj hp.2.1 hq.2.1 eq
  obtain ⟨base,row⟩ := Host.mul_add_inj hp.1 hq.1 base
  obtain ⟨batch,chunk⟩ := Host.mul_add_inj
    (show pid2 p<h.shape.chunks*h.shape.groups from (p ⟨2,by change 2<3; decide⟩).isLt)
    (show pid2 q<h.shape.chunks*h.shape.groups from (q ⟨2,by change 2<3; decide⟩).isLt) base
  obtain ⟨tileRow,laneRow⟩ := Host.mul_add_inj i.isLt i'.isLt row
  obtain ⟨tileCol,laneCol⟩ := Host.mul_add_inj j.isLt j'.isLt col
  refine ⟨Host.pid_ext ?_ batch chunk,Fin.ext laneRow,Fin.ext laneCol⟩
  have d := Nat.div_add_mod (pid0 p) (Host.cdiv h.shape.chunk 64)
  have d' := Nat.div_add_mod (pid0 q) (Host.cdiv h.shape.chunk 64)
  rw [tileRow,tileCol] at d
  exact d.symm.trans d'

#axiomsClean writer_unique
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.Cleanup
