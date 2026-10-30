import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.HostPrelude

namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace HostLayout
open Torch Address Memory Execution
open VeriTile.Bench.Repairs.LayerNormFwd.Address (byteAddress)
noncomputable section
open scoped Classical

structure View3 (t : Tensor) (shape : Shape) where
  (seqStride headStride dimStride : Nat)
  dimensions : t.shape=[shape.sequences,shape.heads,shape.columns]
  strides : t.strides=[seqStride,headStride,dimStride]
structure View2 (t : Tensor) (shape : Shape) where
  (headStride dimStride : Nat)
  dimensions : t.shape=[shape.heads,shape.columns]
  strides : t.strides=[headStride,dimStride]

def view3 (t : Tensor) (shape : Shape) (hs : t.shape=[shape.sequences,shape.heads,shape.columns]) : View3 t shape := by
  have hl : t.strides.length=3 := by rw [t.rank,hs];rfl
  apply Classical.choice
  obtain ⟨a,b,c,h⟩ := List.length_eq_three.mp hl
  exact ⟨⟨a,b,c,hs,h⟩⟩
def view2 (t : Tensor) (shape : Shape) (hs : t.shape=[shape.heads,shape.columns]) : View2 t shape := by
  have hl : t.strides.length=2 := by rw [t.rank,hs];rfl
  apply Classical.choice
  obtain ⟨a,b,h⟩ := List.length_eq_two.mp hl
  exact ⟨⟨a,b,hs,h⟩⟩

def index3 (row : Row shape) (col : Fin shape.columns) : List Nat := [row.1.val,row.2.val,col.val]
def index2 (row : Row shape) (col : Fin shape.columns) : List Nat := [row.2.val,col.val]

theorem coordinate3 (v : View3 t shape) (row : Row shape) (col : Fin shape.columns) :
    Torch.Coordinate t (index3 row col) := by
  unfold Torch.Coordinate
  rw [v.dimensions]
  exact .cons row.1.isLt (.cons row.2.isLt (.cons col.isLt .nil))
theorem coordinate2 (v : View2 t shape) (row : Row shape) (col : Fin shape.columns) :
    Torch.Coordinate t (index2 row col) := by
  unfold Torch.Coordinate
  rw [v.dimensions]
  exact .cons row.2.isLt (.cons col.isLt .nil)

def strides (xv : View3 x shape) (wv : View2 w shape) (yv : View3 y shape) : Strides :=
  ⟨xv.seqStride,xv.headStride,xv.dimStride,yv.seqStride,yv.headStride,yv.dimStride,wv.headStride,wv.dimStride⟩

theorem offset3 (v : View3 t shape) (row : Row shape) (col : Fin shape.columns) :
    (List.zipWith (fun (i n : Nat) => i*n) (index3 row col) t.strides).sum=
      offset row.1.val row.2.val col.val v.seqStride v.headStride v.dimStride := by
  simp [v.strides,index3,offset,Nat.add_assoc]
theorem offset2 (v : View2 t shape) (row : Row shape) (col : Fin shape.columns) :
    (List.zipWith (fun (i n : Nat) => i*n) (index2 row col) t.strides).sum=
      offset 0 row.2.val col.val 0 v.headStride v.dimStride := by
  simp [v.strides,index2,offset]

/-- The mathematical layout is constructed from actual tensor descriptors,
actual stride metadata and this empty_like call's ordinary freshness law. -/
def layout (xv : View3 x shape) (wv : View2 w shape) (yv : View3 y shape)
    (x4 : x.allocation.width=4) (w4 : w.allocation.width=4) (y4 : y.allocation.width=4)
    (injective : ∀ i j, Torch.Coordinate y i → Torch.Coordinate y j →
      address y i=address y j → i=j)
    (freshX : ∀ i, Covers x i → ¬Covers y i)
    (freshW : ∀ i, Covers w i → ¬Covers y i) : Layout shape (strides xv wv yv) where
  x := x.allocation
  w := w.allocation
  y := y.allocation
  xWidth := x4
  wWidth := w4
  yWidth := y4
  xBounds := by
    intro row col
    simpa only [offset3 xv row col,xOffset,strides] using x.bounds _ (coordinate3 xv row col)
  wBounds := by
    intro row col
    simpa only [offset2 wv row col,wOffset,strides] using w.bounds _ (coordinate2 wv row col)
  yBounds := by
    intro row col
    simpa only [offset3 yv row col,yOffset,strides] using y.bounds _ (coordinate3 yv row col)
  yInjective := by
    intro a b he
    have hi : index3 a.1 a.2=index3 b.1 b.2 := injective _ _
      (coordinate3 yv a.1 a.2) (coordinate3 yv b.1 b.2) (by
        simpa only [address,offset3 yv,yOffset,strides] using congrArg (byteAddress y.allocation) he)
    simp only [index3,List.cons.injEq,and_true] at hi
    exact Prod.ext (Prod.ext (Fin.ext hi.1) (Fin.ext hi.2.1)) (Fin.ext hi.2.2)
  xApart := fun i h1 h2 => freshX i ⟨h1,h2⟩
  wApart := fun i h1 h2 => freshW i ⟨h1,h2⟩

theorem x_address (xv : View3 x shape) (wv : View2 w shape) (yv : View3 y shape)
    (x4 w4 y4 inj fx fw) (row : Row shape) (col : Fin shape.columns) :
    xAddress (layout xv wv yv x4 w4 y4 inj fx fw) row col=address x (index3 row col) := by
  simp only [xAddress,layout,address,offset3 xv,xOffset,strides]
theorem w_address (xv : View3 x shape) (wv : View2 w shape) (yv : View3 y shape)
    (x4 w4 y4 inj fx fw) (row : Row shape) (col : Fin shape.columns) :
    wAddress (layout xv wv yv x4 w4 y4 inj fx fw) row col=address w (index2 row col) := by
  simp only [wAddress,layout,address,offset2 wv,wOffset,strides]
theorem y_address (xv : View3 x shape) (wv : View2 w shape) (yv : View3 y shape)
    (x4 w4 y4 inj fx fw) (row : Row shape) (col : Fin shape.columns) :
    yAddress (layout xv wv yv x4 w4 y4 inj fx fw) row col=address y (index3 row col) := by
  simp only [yAddress,layout,address,offset3 yv,yOffset,strides]

theorem caller_outside (layout : Layout shape st) (caller : Nat → Prop)
    (fresh : ∀ i, layout.y.base ≤ i → i < layout.y.base+layout.y.extent*layout.y.width → ¬caller i)
    (i : Nat) (hi : caller i) (row : Row shape) (col : Fin shape.columns) :
    i < yAddress layout row col ∨ yAddress layout row col+4 ≤ i := by
  obtain ⟨hlo,hhi⟩ := cell_span layout.y layout.yWidth (layout.yBounds row col)
  by_contra h
  have hb : layout.y.base ≤ i ∧ i < layout.y.base+layout.y.extent*layout.y.width := by
    change layout.y.base ≤ byteAddress layout.y (yOffset st row col.val) at hlo
    change byteAddress layout.y (yOffset st row col.val)+4 ≤ _ at hhi
    unfold yAddress at h
    omega
  exact fresh i hb.1 hb.2 hi

#axiomsClean layout
#axiomsClean caller_outside
#guard_msgs (drop info) in
#auditModuleAxioms
end
end HostLayout
end VeriTile.Bench.Repairs.LayernormFwdTriton
