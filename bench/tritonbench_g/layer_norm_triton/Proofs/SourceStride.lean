import bench.tritonbench_g.layer_norm_triton.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceStride
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
noncomputable section

/-- A contiguous matrix's observed row stride has the required effect on
every launched row, including singleton rows with arbitrary stored strides. -/
theorem dense_row_offset (d : Dense) (M N row : Nat)
    (shape : d.tensor.view.shape=[M,N]) (hn : 0<N) (hr : row<M) :
    row*d.tensor.view.strides.headD 0=row*N := by
  have hl : d.tensor.view.strides.length=2 := by rw [d.tensor.view.rank,shape]; rfl
  obtain ⟨a,b,hs⟩ := List.length_eq_two.mp hl
  have hcount : d.count=M*N := by rw [d.countEq,shape]; simp
  have hi : row*N<d.count := by rw [hcount]; exact Nat.mul_lt_mul_of_pos_right hr hn
  have hd : row*N/N=row := Nat.mul_div_cancel row hn
  have hm : row*N%N=0 := by simp
  have ha := d.coordinates (row*N) hi
  simp only [shape,coordinates,List.prod_cons,List.prod_nil,Nat.mul_one,hd,Nat.mul_mod_right,
    Nat.mod_one,Nat.div_one,Torch.address,VeriTile.Bench.Repairs.LayerNormFwd.Address.byteAddress,
    hs,List.zipWith_cons_cons,List.zipWith_nil_left,List.sum_cons,List.sum_nil,Nat.zero_mul,Nat.add_zero,
    d.tensor.view.fp32Width d.tensor.fp32] at ha
  simp only [hd,hm,Nat.zero_mul,Nat.add_zero] at ha
  simp only [hs,List.headD_cons]
  omega

#axiomsClean dense_row_offset
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceStride
