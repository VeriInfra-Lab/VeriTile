import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Grid
namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 3000000

def DenseView (rows cols sr sc capacity : Nat) : Prop :=
  rows*cols≤capacity ∧ ∀ r c,r<rows → c<cols → r*sr+c*sc=r*cols+c

theorem dense_bound (rows cols sr sc capacity : Nat)
    (h : DenseView rows cols sr sc capacity) (r c : Nat) (hr : r<rows) (hc : c<cols) :
    r*sr+c*sc<capacity := by
  rw [h.2 r c hr hc]
  calc
    r*cols+c < r*cols+cols := Nat.add_lt_add_left hc _
    _ = (r+1)*cols := by ring
    _ ≤ rows*cols := Nat.mul_le_mul_right _ (by omega)
    _ ≤ capacity := h.1

theorem dense_injective (rows cols sr sc capacity : Nat)
    (h : DenseView rows cols sr sc capacity) :
    Function.Injective (fun x : Fin rows × Fin cols => x.1.val*sr+x.2.val*sc) := by
  intro x y he
  dsimp only at he
  rw [h.2 _ _ x.1.isLt x.2.isLt,h.2 _ _ y.1.isLt y.2.isLt] at he
  have hcols : 0<cols := by have := x.2.isLt; omega
  have hr : x.1.val=y.1.val := by
    have hx := Nat.add_mul_div_left x.2.val x.1.val hcols
    have hy := Nat.add_mul_div_left y.2.val y.1.val hcols
    rw [Nat.div_eq_of_lt x.2.isLt] at hx
    rw [Nat.div_eq_of_lt y.2.isLt] at hy
    rw [Nat.add_comm,Nat.mul_comm] at hx hy
    rw [he] at hx
    omega
  have hc : x.2.val=y.2.val := by rw [hr] at he; omega
  exact Prod.ext (Fin.ext hr) (Fin.ext hc)

theorem active_a_bound {α : Type} (p : Parameters) (s : State α) (cap block : Nat)
    (h : DenseView p.rows (inner p) p.inputRow p.inputInner cap)
    (i : AIndex p) (hr : row p s i.1<p.rows)
    (hk : i.2.1.val< inner p-block*p.blockInner) : aOffset p s block i<cap := by
  have hglobal : block*p.blockInner+i.2.1.val< inner p := by omega
  have he : aOffset p s block i=
      row p s i.1*p.inputRow+(block*p.blockInner+i.2.1.val)*p.inputInner := by
    simp only [aOffset]; ring
  rw [he]
  exact dense_bound _ _ _ _ _ h _ _ hr hglobal

theorem active_b_bound {α : Type} (p : Parameters) (s : State α) (cap block : Nat)
    (h : DenseView (inner p) p.columns p.weightInner p.weightColumn cap)
    (i : BIndex p) (hk : i.1.val< inner p-block*p.blockInner)
    (hc : column p s i.2.1<p.columns) : bOffset p s block i<cap := by
  have hglobal : block*p.blockInner+i.1.val< inner p := by omega
  have he : bOffset p s block i=
      (block*p.blockInner+i.1.val)*p.weightInner+column p s i.2.1*p.weightColumn := by
    simp only [bOffset]; ring
  rw [he]
  exact dense_bound _ _ _ _ _ h _ _ hglobal hc

theorem active_c_bound {α : Type} (p : Parameters) (s : State α) (cap : Nat)
    (h : DenseView p.rows p.columns p.outputRow p.outputColumn cap)
    (i : CIndex p) (hi : active p s i) : cOffset p s i<cap :=
  dense_bound _ _ _ _ _ h _ _ hi.1 hi.2

#axiomsClean dense_bound
#axiomsClean dense_injective
#axiomsClean active_a_bound
#axiomsClean active_b_bound
#axiomsClean active_c_bound
end
end VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Core
