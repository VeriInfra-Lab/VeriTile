import bench.repairs.tritonbench_g.fast_rms_layernorm.Kernels

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.Geometry
open VeriTile.Triton

def featureCount (h : Call) : Nat := (h.cols+255)/256
def column (pid : Nat) (j : Fin 256) : Nat := pid*256+j.val

theorem feature_limit (h : Call) : 0<featureCount h ∧ featureCount h≤256 := by
  have p:=h.positive
  have c:=h.covered
  have b:=h.blockLimit
  unfold featureCount
  omega

theorem column_limit (h : Call) (p : Fin (featureCount h)) (j : Fin 256) : column p.val j<65536 := by
  have f:=feature_limit h
  have hp:=p.isLt
  have hj:=j.isLt
  unfold column
  omega

def owner (h : Call) (j : Fin h.cols) : Fin (featureCount h) :=
  ⟨j.val/256,by have hj:=j.isLt; unfold featureCount; omega⟩
def lane (j : Fin N) : Fin 256 := ⟨j.val%256,Nat.mod_lt _ (by decide)⟩

theorem covered (h : Call) (j : Fin h.cols) : column (owner h j).val (lane j)=j.val := by
  simp [column,owner,lane,Nat.div_add_mod]
  omega

theorem column_injective (p q : Nat) (i j : Fin 256) (eq : column p i=column q j) : p=q ∧ i=j := by
  have hi:=i.isLt
  have hj:=j.isLt
  unfold column at eq
  exact ⟨by omega,Fin.ext (by omega)⟩

theorem row_injective (h : Call) (r s : Nat) (i j : Fin h.cols)
    (eq : r*h.cols+i.val=s*h.cols+j.val) : r=s ∧ i=j := by
  have re : (r*h.cols+i.val)/h.cols=r := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ h.positive,Nat.div_eq_of_lt i.isLt,Nat.zero_add]
  have se : (s*h.cols+j.val)/h.cols=s := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ h.positive,Nat.div_eq_of_lt j.isLt,Nat.zero_add]
  have rs : r=s := re.symm.trans ((congrArg (·/h.cols) eq).trans se)
  exact ⟨rs,Fin.ext (by subst s; omega)⟩

#axiomsClean covered
#axiomsClean row_injective
#axiomsClean column_injective
end VeriTile.Bench.Repairs.FastRmsLayernorm.Geometry
