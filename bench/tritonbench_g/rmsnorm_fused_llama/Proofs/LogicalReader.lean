import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.RealPublicIO

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.RealPublicIO
open VeriTile Triton
open scoped VeriTile.Triton.LogicalScheduledKernelIO
noncomputable section

/-- Every logical matrix read is the same finite row-major input cell. -/
theorem matrixValues_apply (rows N : Nat) (xs : Fin (rows*N) → ℝ)
    (row : Fin rows) (col : Fin N) :
    IO.matrixValues rows N xs row col = IO.vector xs (row.val*N+col.val) := by
  have bound : row.val*N+col.val<rows*N := by
    calc row.val*N+col.val < (row.val+1)*N := by nlinarith [col.isLt]
         _ ≤ rows*N := Nat.mul_le_mul_right N row.isLt
  simp only [IO.matrixValues,Equiv.coe_fn_mk,IO.vector,dif_pos bound]
  congr 1
  apply Fin.ext
  simp [finProdFinEquiv,Nat.mul_comm,Nat.add_comm]

/-- Complete old/new judgments are equivalent, including execution and frame. -/
theorem logical_iff (candidate : Bool) (X Y W : RegionName) (rows N B : Nat) (epsilon : ℝ) :
    (IO.logicalIO candidate X Y W rows N B epsilon ⊨ fun values cell =>
      let xs := values.1
      let weights := values.2
      let row := cell.1
      let col := cell.2
      xs row col * (Real.sqrt ((∑ k : Fin N, (xs row k)^2)/(N:ℝ)+epsilon))⁻¹ * weights col) ↔
    (IO.grid candidate X Y W rows N B epsilon).RegionalScheduledImplements
      (fun _ => IO.schedule rows rows) (Reference.real candidate X Y W rows N B epsilon) := by
  change (IO.grid candidate X Y W rows N B epsilon).RegionalScheduledImplements _ _ ↔ _
  have target_eq : (fun a (_ : Fin 1) (j : Fin (rows*N)) =>
      IO.matrixValues rows N (a (0 : Fin 2)) (finProdFinEquiv.symm j).1 (finProdFinEquiv.symm j).2 *
        (Real.sqrt ((∑ k : Fin N,
          (IO.matrixValues rows N (a (0 : Fin 2)) (finProdFinEquiv.symm j).1 k)^2)/(N:ℝ)+epsilon))⁻¹ *
        a (1 : Fin 2) (finProdFinEquiv.symm j).2) =
      Reference.real candidate X Y W rows N B epsilon := by
    funext a o j
    simp only [matrixValues_apply,Reference.real,RealGrid.value]
    simp only [finProdFinEquiv_symm_apply]
    have positive : 0<N := by nlinarith [j.isLt]
    congr 1
    change a (1 : Fin 2) j.modNat = (if h : j.val%N<N then a (1 : Fin 2) ⟨j.val%N,h⟩ else (0 : ℝ))
    rw [dif_pos (Nat.mod_lt j.val positive)]
    apply congrArg (a (1 : Fin 2))
    apply Fin.ext
    rfl
  exact congrArg (fun f => (IO.grid candidate X Y W rows N B epsilon).RegionalScheduledImplements
    (fun _ => IO.schedule rows rows) f) target_eq |>.to_iff

end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.RealPublicIO
