import bench.repairs.tritonbench_g.diag_ssm_triton.IO

namespace VeriTile.Bench.Repairs.DiagSsmTriton.Interface
open VeriTile.Triton
noncomputable section

/-- Row-major indexing of finite logical channels and time/lane coordinates. -/
def pair {m n : Nat} (i : Fin m) (j : Fin n) : Fin (m*n) :=
  ⟨i.val*n+j.val,calc
    i.val*n+j.val < i.val*n+n := Nat.add_lt_add_left j.isLt _
    _ = (i.val+1)*n := by simp [Nat.add_mul]
    _ ≤ m*n := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt i.isLt)⟩

theorem pair_val {m n : Nat} (i : Fin m) (j : Fin n) :
    (pair i j).val = i.val*n+j.val := rfl

@[simp] theorem pair_div {m n : Nat} (i : Fin m) (j : Fin n) :
    (pair i j).val/n=i.val := by
  have hn : 0<n := Nat.zero_lt_of_lt j.isLt
  change (i.val*n+j.val)/n=i.val
  rw [Nat.mul_comm i.val n,Nat.mul_add_div hn]
  simp [Nat.div_eq_of_lt j.isLt]

@[simp] theorem pair_mod {m n : Nat} (i : Fin m) (j : Fin n) :
    (pair i j).val%n=j.val := by simp [pair,Nat.add_mod,Nat.mod_eq_of_lt j.isLt]

@[simp] theorem divNat_pair {m n : Nat} (i : Fin m) (j : Fin n) :
    (pair i j).divNat=i := Fin.ext (pair_div i j)

@[simp] theorem modNat_pair {m n : Nat} (i : Fin m) (j : Fin n) :
    (pair i j).modNat=j := Fin.ext (pair_mod i j)

@[simp] theorem streamAddress_pair (batch dim B pid length : Nat) (t : Fin length) (j : Fin B) :
    streamAddress batch dim B pid (pair t j).val=t.val*(batch*dim)+column B pid j.val := by
  simp only [streamAddress,pair_div,pair_mod]

@[simp] theorem extend_at {n : Nat} (v : Fin n → ℝ) (k : Fin n) : extend v k.val=v k := by
  simp only [extend,k.isLt,↓reduceDIte]

end
end VeriTile.Bench.Repairs.DiagSsmTriton.Interface
