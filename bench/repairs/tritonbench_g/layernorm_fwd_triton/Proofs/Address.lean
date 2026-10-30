import bench.repairs.tritonbench_g.layernorm_fwd_triton.Kernels
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.Address

/-! Guarded feature/row addresses. The output allocation bounds the loop
induction, including its final increment. No signed-i32 shape restriction or
positive-N precondition is imposed. Inactive pointer products may wrap; masked
lanes never access them. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Address
open VeriTile.Bench.Repairs.LayerNormFwd.Address
noncomputable section
open scoped Classical

abbrev Allocation := VeriTile.Bench.Repairs.LayerNormFwd.Address.Allocation
abbrev Row (shape : Shape) := Fin shape.sequences × Fin shape.heads
abbrev Coordinate (shape : Shape) := Row shape × Fin shape.columns

def offset (s h col strideS strideH strideD : Nat) : Nat :=
  s*strideS+h*strideH+col*strideD

def xOffset (strides : Strides) (row : Row shape) (col : Nat) : Nat :=
  offset row.1.val row.2.val col strides.xSeq strides.xHead strides.xDim
def yOffset (strides : Strides) (row : Row shape) (col : Nat) : Nat :=
  offset row.1.val row.2.val col strides.ySeq strides.yHead strides.yDim
def wOffset (strides : Strides) (row : Row shape) (col : Nat) : Nat :=
  offset 0 row.2.val col 0 strides.wHead strides.wDim

def wordOffset (s h col strideS strideH strideD : Nat) : Nat :=
  add64 (add64 (mul64 (wrap64 s) (wrap64 strideS))
    (mul64 (wrap64 h) (wrap64 strideH))) (mul64 (wrap64 col) (wrap64 strideD))

theorem word_mod (s h col strideS strideH strideD : Nat) :
    wordOffset s h col strideS strideH strideD=offset s h col strideS strideH strideD % 2^64 := by
  simp [wordOffset,offset,add64,mul64,wrap64,Nat.add_mod,Nat.mul_mod]

theorem guarded_word_exact (s h col strideS strideH strideD : Nat) (a : Allocation)
    (bound : offset s h col strideS strideH strideD < a.extent) :
    wordOffset s h col strideS strideH strideD=offset s h col strideS strideH strideD  ∧ 
    signed64 (wordOffset s h col strideS strideH strideD)=
      (offset s h col strideS strideH strideD:ℤ) := by
  have b : offset s h col strideS strideH strideD < 2^63 := lt_of_lt_of_le bound a.signed_extent
  rw [word_mod,Nat.mod_eq_of_lt (lt_trans b (by norm_num))]
  exact ⟨rfl,signed64_exact b⟩

/-- Local tensor-library metadata, bounds, and empty_like injectivity. Alias
between X and W is admitted. Freshness concerns the complete byte windows. -/
structure Layout (shape : Shape) (strides : Strides) where
  (x w y : Allocation)
  (xWidth : x.width=4) (wWidth : w.width=4) (yWidth : y.width=4)
  xBounds : ∀ row : Row shape, ∀ col : Fin shape.columns, xOffset strides row col.val < x.extent
  wBounds : ∀ row : Row shape, ∀ col : Fin shape.columns, wOffset strides row col.val < w.extent
  yBounds : ∀ row : Row shape, ∀ col : Fin shape.columns, yOffset strides row col.val < y.extent
  yInjective : Function.Injective (fun q : Coordinate shape => yOffset strides q.1 q.2.val)
  xApart : ∀ i, x.base ≤ i → i < x.base+x.extent*x.width →
    ¬(y.base ≤ i  ∧  i < y.base+y.extent*y.width)
  wApart : ∀ i, w.base ≤ i → i < w.base+w.extent*w.width →
    ¬(y.base ≤ i  ∧  i < y.base+y.extent*y.width)

def xAddress (layout : Layout shape strides) (row : Row shape) (col : Fin shape.columns) : Nat :=
  byteAddress layout.x (xOffset strides row col.val)
def wAddress (layout : Layout shape strides) (row : Row shape) (col : Fin shape.columns) : Nat :=
  byteAddress layout.w (wOffset strides row col.val)
def yAddress (layout : Layout shape strides) (row : Row shape) (col : Fin shape.columns) : Nat :=
  byteAddress layout.y (yOffset strides row col.val)

theorem y_address_injective (layout : Layout shape strides) :
    Function.Injective (fun q : Coordinate shape => yAddress layout q.1 q.2) := by
  intro a b h
  apply layout.yInjective
  unfold yAddress byteAddress at h
  have hh := Nat.add_left_cancel h
  exact Nat.eq_of_mul_eq_mul_right layout.y.width_pos hh

theorem input_ne_output (layout : Layout shape strides) (r q : Row shape)
    (j k : Fin shape.columns) :
    xAddress layout r j ≠ yAddress layout q k  ∧  wAddress layout r j ≠ yAddress layout q k := by
  have hx := byte_address layout.x (layout.xBounds r j)
  have hw := byte_address layout.w (layout.wBounds r j)
  have hy := byte_address layout.y (layout.yBounds q k)
  constructor
  · intro h
    apply layout.xApart _ hx.1 hx.2.1
    change layout.y.base ≤ xAddress layout r j ∧ xAddress layout r j < layout.y.base+layout.y.extent*layout.y.width
    rw [h]
    exact ⟨hy.1,hy.2.1⟩
  · intro h
    apply layout.wApart _ hw.1 hw.2.1
    change layout.y.base ≤ wAddress layout r j ∧ wAddress layout r j < layout.y.base+layout.y.extent*layout.y.width
    rw [h]
    exact ⟨hy.1,hy.2.1⟩

/-- The complete successful output allocation, not an input view stride,
forces the column count to fit its allocated element span. -/
theorem columns_le_extent (layout : Layout shape strides) (row : Row shape) :
    shape.columns ≤ layout.y.extent := by
  let embed : Fin shape.columns → Fin layout.y.extent :=
    fun j => ⟨yOffset strides row j.val,layout.yBounds row j⟩
  have hi : Function.Injective embed := by
    intro a b h
    have hp : (row,a)=(row,b) := layout.yInjective (congrArg Fin.val h)
    exact congrArg Prod.snd hp
  simpa using Fintype.card_le_of_injective embed hi

theorem allocation_loop_bound (layout : Layout shape strides) (row : Row shape) :
    shape.columns+128 < 2^63 := by
  have hn := columns_le_extent layout row
  have hy := layout.y.bytes_finite
  rw [layout.yWidth] at hy
  omega

/-- Every actually executed loop offset/lane addition and final increment
is signed64-exact, independently of masked pointer products. -/
theorem loop_exact (layout : Layout shape strides) (row : Row shape)
    (off : Nat) (active : off < shape.columns) (lane : Fin 128) :
    signed64 (off+lane.val)=(off+lane.val:Nat)  ∧ 
    signed64 (off+128)=(off+128:Nat) := by
  have hb := allocation_loop_bound layout row
  exact ⟨signed64_exact (by omega),signed64_exact (by omega)⟩

#axiomsClean guarded_word_exact
#axiomsClean input_ne_output
#axiomsClean allocation_loop_bound
#axiomsClean loop_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end Address
end VeriTile.Bench.Repairs.LayernormFwdTriton
