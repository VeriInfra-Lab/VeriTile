import bench.tritonbench_g.matmul_dequantize_int4.Kernels
import bench.tritonbench_g.matmul_dequantize_int4.Proofs.Grid
import bench.tritonbench_g.matmul_kernel.Proofs.Geometry
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4
theorem tuning_facts (i : Fin 17) :
    0<(tunings i).BM ∧ (tunings i).BM≤128 ∧
    0<(tunings i).BN ∧ (tunings i).BN∣256 ∧
    0<(tunings i).BK ∧ (tunings i).BK∣128 ∧
    16∣(tunings i).BK := by fin_cases i <;> decide +kernel

theorem ceil_exact (x b : Nat) (hb : 0<b) (hdiv : b∣x) :
    (x+b-1)/b=x/b := by
  obtain ⟨q,rfl⟩ := hdiv
  have he : b*q+b-1=(b-1)+b*q := by omega
  rw [he,Nat.add_mul_div_left _ _ hb,Nat.div_eq_of_lt (by omega : b-1<b)]
  simp [Nat.mul_div_right _ hb]

theorem legal (c : Call) (h : c.Valid) (i : Fin 17) : Logical.Legal (params c i) := by
  have ht := tuning_facts i
  have hbk : (tunings i).BK∣c.K := dvd_trans ht.2.2.2.2.2.1 h.k128
  have hbg : (tunings i).BK∣c.group := dvd_trans ht.2.2.2.2.2.1 h.g128
  refine ⟨ht.2.2.2.2.1,?_,?_,Nat.div_mul_cancel hbg,h.g,rfl⟩
  · exact Nat.div_mul_cancel (dvd_trans (by decide : 8∣16) ht.2.2.2.2.2.2)
  · change ((c.K+(tunings i).BK-1)/(tunings i).BK)*(tunings i).BK=c.K
    rw [ceil_exact c.K _ ht.2.2.2.2.1 hbk]
    exact Nat.div_mul_cancel hbk

theorem geometry_valid (c : Call) (h : c.Valid) (i : Fin 17) :
    VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry.Valid (Grid.geometry (params c i)) := by
  refine ⟨(tuning_facts i).1,(tuning_facts i).2.2.1,by change 0<8; decide,?_⟩
  intro x y he
  have eq : (x.1.val,x.2.val)=(y.1.val,y.2.val) :=
    MatmulKernel.view_injective c.C h.cValid _ _
      (by rw [h.cShape.1,h.cShape.2]; exact ⟨x.1.isLt,x.2.isLt⟩)
      (by rw [h.cShape.1,h.cShape.2]; exact ⟨y.1.isLt,y.2.isLt⟩) he
  exact Prod.ext (Fin.ext (congrArg Prod.fst eq)) (Fin.ext (congrArg Prod.snd eq))

#axiomsClean tuning_facts
#axiomsClean legal
#axiomsClean geometry_valid
end VeriTile.Bench.Repairs.MatmulDequantizeInt4
