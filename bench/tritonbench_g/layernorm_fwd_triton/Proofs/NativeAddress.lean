import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Address

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Address
open VeriTile.Bench.Repairs.LayerNormFwd.Address
noncomputable section

/-- Native pointer construction after the source's pre-operation widening. -/
def nativePointer (a : Allocation) (s h col strideS strideH strideD : Nat) : Nat :=
  add64 (wrap64 a.base) (mul64 (wordOffset s h col strideS strideH strideD) a.width)

theorem native_pointer_exact (a : Allocation) (s h col strideS strideH strideD : Nat)
    (bound : offset s h col strideS strideH strideD < a.extent) :
    nativePointer a s h col strideS strideH strideD=
      byteAddress a (offset s h col strideS strideH strideD) := by
  have offsetExact := (guarded_word_exact s h col strideS strideH strideD a bound).1
  have byteBound := (byte_address a bound).2.2.1
  change a.base+offset s h col strideS strideH strideD*a.width < 2^64 at byteBound
  have baseBound : a.base < 2^64 := by omega
  have productBound : offset s h col strideS strideH strideD*a.width < 2^64 := by omega
  simp only [nativePointer,offsetExact,add64,mul64,wrap64,Nat.mod_eq_of_lt baseBound,
    Nat.mod_eq_of_lt productBound,Nat.mod_eq_of_lt byteBound,byteAddress]

theorem native_x (layout : Layout shape strides) (row : Row shape) (col : Fin shape.columns) :
    nativePointer layout.x row.1.val row.2.val col.val strides.xSeq strides.xHead strides.xDim=
      xAddress layout row col :=
  native_pointer_exact _ _ _ _ _ _ _ (layout.xBounds row col)

theorem native_w (layout : Layout shape strides) (row : Row shape) (col : Fin shape.columns) :
    nativePointer layout.w 0 row.2.val col.val 0 strides.wHead strides.wDim=wAddress layout row col :=
  native_pointer_exact _ _ _ _ _ _ _ (layout.wBounds row col)

theorem native_y (layout : Layout shape strides) (row : Row shape) (col : Fin shape.columns) :
    nativePointer layout.y row.1.val row.2.val col.val strides.ySeq strides.yHead strides.yDim=
      yAddress layout row col :=
  native_pointer_exact _ _ _ _ _ _ _ (layout.yBounds row col)

/-- Loop strength reduction is word arithmetic: even an inactive pointer
increment may wrap. The next active address still equals its direct formula. -/
theorem pointer_increment (base step off : Nat) :
    add64 (wrap64 (base+off*step)) (wrap64 (128*step))=
      wrap64 (base+(off+128)*step) := by
  simp only [add64,wrap64]
  rw [←Nat.add_mod]
  congr 1
  simp [Nat.add_mul,Nat.add_assoc]

#axiomsClean native_pointer_exact
#axiomsClean native_x
#axiomsClean native_w
#axiomsClean native_y
#axiomsClean pointer_increment
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Address
