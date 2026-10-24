import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.KernelBytes

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

/-- Independent backward polynomial on actual logical Tensor inputs. -/
def realValue (c : Call) (before : Nat→ℝ) (key : Bool) (row head col : Nat) : ℝ :=
  if col<2*c.geometry.half then
    let j:=if col<c.geometry.half then col else col-c.geometry.half
    let pair:=Spec.pair
      (logicalRead before (c.q key) (c.geometry.offset key row head j))
      (logicalRead before (c.q key) (c.geometry.offset key row head (j+c.geometry.half)))
      (logicalRead before (c.tables Bool.false) (row%c.geometry.sequence*c.geometry.half+j))
      (logicalRead before (c.tables Bool.true) (row%c.geometry.sequence*c.geometry.half+j))
    if col<c.geometry.half then pair.1 else pair.2
  else logicalRead before (c.q key) (c.geometry.offset key row head col)

/-- Independent native graph: low=fma(c,a,mul(s,b)),
high=fma(c,b,neg(mul(s,a))). Both same-FP32 casts remain in the source model. -/
def nativeValue (p : Numerics.Primitives) (c : Call) (before : Nat→Word)
    (key : Bool) (row head col : Nat) : Word :=
  if col<2*c.geometry.half then
    let j:=if col<c.geometry.half then col else col-c.geometry.half
    Numerics.pair p (decide (c.geometry.half≤col))
      (logicalRead before (c.q key) (c.geometry.offset key row head j))
      (logicalRead before (c.q key) (c.geometry.offset key row head (j+c.geometry.half)))
      (logicalRead before (c.tables Bool.false) (row%c.geometry.sequence*c.geometry.half+j))
      (logicalRead before (c.tables Bool.true) (row%c.geometry.sequence*c.geometry.half+j))
  else logicalRead before (c.q key) (c.geometry.offset key row head col)

theorem real_binding (c : Call) (before : Nat→ℝ) (key : Bool) (row head col : Nat) :
    tensorScalar ideal id c before key row head col=realValue c before key row head col := by
  unfold tensorScalar realValue
  split
  · dsimp only
    rw [ideal_pair]
    by_cases low : col<c.geometry.half <;>
      simp [low,show (c.geometry.half≤col)↔¬col<c.geometry.half from by omega]
  · rfl

theorem native_binding (p : Numerics.Primitives) (c : Call) (before : Nat→Word)
    (key : Bool) (row head col : Nat) :
    (tensorScalar (Numerics.operations p) Numerics.leaf c before key row head col).word=
      nativeValue p c before key row head col := by
  unfold tensorScalar nativeValue
  split
  · exact Numerics.source_pair p _ _ _ _ _
  · rfl

theorem native_bytes (p : Numerics.Primitives) (c : Call) (admit : Admitted c) (before : Bytes)
    (key : Bool) (row head col : Nat) (hr : row<c.geometry.rows) (hh : head<c.geometry.heads key)
    (hc : col<c.geometry.dimension) (byte : Fin 4) :
    byteMemory c before (grid (Numerics.operations p).backend c.geometry
        (memory Numerics.leaf (read32 before) c) (launchRows c.geometry))
        ((c.q key).storage.address ((c.q key).physicalIndex (c.geometry.offset key row head col))+byte.val)=
      wordByte (nativeValue p c (read32 before) key row head col) byte := by
  rw [tensor_bytes c admit before _ key row head col hr hh hc byte,
    grid_values (Numerics.operations p) Numerics.leaf c admit (read32 before) key row head col hr hh hc,
    native_binding]
  split
  · rfl
  · rename_i tail
    simp only [nativeValue,if_neg tail,logicalRead]
    exact (read32_byte before _ byte).symm

theorem native_words (p : Numerics.Primitives) (c : Call) (admit : Admitted c) (before : Bytes)
    (key : Bool) (row head col : Nat) (hr : row<c.geometry.rows) (hh : head<c.geometry.heads key)
    (hc : col<c.geometry.dimension) :
    logicalRead (read32 (byteMemory c before (grid (Numerics.operations p).backend c.geometry
        (memory Numerics.leaf (read32 before) c) (launchRows c.geometry))))
      (c.q key) (c.geometry.offset key row head col)=
      nativeValue p c (read32 before) key row head col := by
  have bytes:=native_bytes p c admit before key row head col hr hh hc
  have b0:=bytes ⟨0,by decide⟩
  have b1:=bytes ⟨1,by decide⟩
  have b2:=bytes ⟨2,by decide⟩
  have b3:=bytes ⟨3,by decide⟩
  simp only [Nat.add_zero] at b0
  unfold logicalRead
  change (⟨_ ++ _ ++ _ ++ _⟩ : Word)=_
  rw [b3,b2,b1,b0]
  exact decode_encode _

#axiomsClean real_binding
#axiomsClean native_binding
#axiomsClean native_bytes
#axiomsClean native_words
#specNonCircular realValue avoiding [Call.source,Execution,Source.kernel,grid,physical]
#specNonCircular nativeValue avoiding [Call.source,Execution,Source.kernel,grid,byteMemory]
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
