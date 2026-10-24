import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.HostContract

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.Checked
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding.Memory (API)
set_option maxHeartbeats 3000000

theorem output_bound (r : Request) (k : Bool) (row h j : Nat)
    (hr : row<r.rows) (hh : h<r.paddedHeads k) (hj : j<r.paddedDim/2) :
    row*(r.heads k*r.dimension)+h*r.dimension+j+r.half≤outputLast r k := by
  have h1 := Nat.mul_le_mul_right (r.heads k*r.dimension) (show row≤r.rows-1 by omega)
  have h2 := Nat.mul_le_mul_right r.dimension (show h≤r.paddedHeads k-1 by omega)
  have h3 : j≤r.paddedDim/2-1 := by omega
  unfold outputLast
  omega

theorem table_bound (r : Request) (rowStride colStride row j : Nat)
    (hr : row<r.rows) (hj : j<r.paddedDim/2) :
    row%r.sequence*rowStride+j*colStride≤tableLast r rowStride colStride := by
  have nonzero : 0<r.sequence := by
    by_contra hn
    have hz : r.sequence=0 := by omega
    simp [Request.rows,hz] at hr
  have hm := Nat.mod_lt row nonzero
  have h1 := Nat.mul_le_mul_right rowStride (show row%r.sequence≤r.sequence-1 by omega)
  have h2 := Nat.mul_le_mul_right colStride (show j≤r.paddedDim/2-1 by omega)
  unfold tableLast
  omega

theorem indexABI {α β Store Byte : Type} [Inhabited α] (api : API β Store Byte)
    (embed : β→α) (r : Request) (m : Source.Memory α) (call : Native.Preparation api embed r m)
    (accepted : check api embed r m call=Bool.true) : Native.IndexABI api embed r m call := by
  simp only [check,Bool.and_eq_true] at accepted
  have h := accepted.2
  simp only [launchCheck,Bool.and_eq_true,decide_eq_true_eq] at h
  intro rows half heads
  have on : launched r := ⟨rows,half,heads⟩
  obtain ⟨dim,other⟩ := h.1.1
  have offsets := h.1.2 on
  have pointers := h.2 on
  refine ⟨dim.1,dim.2.1,dim.2.2.1,dim.2.2.2.1,dim.2.2.2.2.1,
    dim.2.2.2.2.2.1,dim.2.2.2.2.2.2.1,dim.2.2.2.2.2.2.2.1,dim.2.2.2.2.2.2.2.2,?_⟩
  intro k
  refine ⟨(other k).1,(other k).2.1,(other k).2.2,?_⟩
  intro row hr head hh j hj
  have output := output_bound r k row head j hr hh hj
  have cosine := table_bound r r.cosRowStride r.cosColStride row j hr hj
  have sine := table_bound r r.sinRowStride r.sinColStride row j hr hj
  have offset := (offsets.1 k)
  have pointer := (pointers.1 k)
  exact ⟨by omega,by omega,by omega,by omega,by omega,by omega⟩

#axiomsClean output_bound
#axiomsClean table_bound
#axiomsClean indexABI
#specNonCircular check avoiding [Native.entry,Memory.run,scalarResult,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RopeBackwardTransform.Checked
