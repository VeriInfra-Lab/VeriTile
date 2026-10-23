import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.KernelTensor
import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.RealWhole
import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.NativeWhole

namespace VeriTile.Bench.Repairs.MatmulDequantize.KernelResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open ByteBridge
noncomputable section
open scoped Classical

def realValue (out : Host.Tensor) (t : TensorBinding.Tensor) (half : Bool)
    (state : State ℝ) (r n : Nat) : ℝ :=
  RealWhole.floatRead half (state.mem t.storageName
    (t.physicalIndex (r*out.columns+n)-t.storageOffset))

/-- Every mathematical output coordinate and the entire exterior typed frame.
The Tensor observation is linked to its actual storage/index binding. -/
def RealResult (out : Host.Tensor) (t : TensorBinding.Tensor) (g : Geometry.Layout)
    (half : Bool) (initial final : State ℝ) (reference : Geometry.Cell g → ℝ) : Prop :=
  (∀ x, realValue out t half final x.1.val x.2.val = reference x) ∧
  (∀ r o, (r≠out.region ∨ ∀ x : Geometry.Cell g, o≠Geometry.address g x) →
    final.mem r o=initial.mem r o)

theorem real_complete (out : Host.Tensor) (t : TensorBinding.Tensor)
    (bound : TensorBinding.Binding out t) (g : Geometry.Layout)
    (shape : out.rows=g.rows ∧ out.columns=g.columns)
    (strides : out.rowStride=g.rowStride ∧ out.columnStride=g.columnStride)
    (half : Bool) (initial final : State ℝ) (reference : Geometry.Cell g → ℝ)
    (values : ∀ x, RealWhole.floatRead half (final.mem out.region (Geometry.address g x))=reference x)
    (frame : ∀ r o, (r≠out.region ∨ ∀ x : Geometry.Cell g, o≠Geometry.address g x) →
      final.mem r o=initial.mem r o) : RealResult out t g half initial final reference := by
  refine ⟨?_,frame⟩
  intro x
  have coord := bound.coordinates x.1.val x.2.val
    (by simpa only [shape.1] using x.1.isLt) (by simpa only [shape.2] using x.2.isLt)
  simp only [realValue,bound.name,bound.offset,coord,strides.1,strides.2]
  rw [show out.storageOffset+x.1.val*g.rowStride+x.2.val*g.columnStride-out.storageOffset=
    Geometry.address g x by unfold Geometry.address;omega]
  exact values x

def logicalIndex (out : Host.Tensor) (g : Geometry.Layout) (x : Grid.Index g) : Nat :=
  Geometry.outRow g x.val.1 x.val.2*out.columns+Geometry.outColumn g x.val.1 x.val.2

/-- Complete native result under one explicit scalar/instruction ISA and one
compiler routing plan. The serialization reads executed values; `expected`
uses only the independent input reference. Primitive semantics and compiler
realization are model boundaries, not deferred numerical experiments. -/
def NativeResult (out : Host.Tensor) (t : TensorBinding.Tensor) (g : Geometry.Layout)
    (width : Nat) (half : Bool) (plan : Native.Plan) (isa : Native.WordISA)
    (before : Memory) (initial final : State Native.Expr) (reference : Grid.Index g → Native.Expr) : Prop :=
  (∀ x : Grid.Index g,final.mem out.region (Grid.address g x)=floatCell half (reference x)) ∧
  serialize out g width half plan isa before final=NativeWhole.expected out g width plan isa before reference ∧
  (∀ x : Store g width, KernelTensor.logicalByte t width
      (serialize out g width half plan isa before final) (logicalIndex out g x.1) x.2.val=
    ((Native.lower plan (reference x.1) x.1.val.1).eval isa).extractLsb' (8*x.2.val) 8) ∧
  (∀ r b, Outside out g width r b → serialize out g width half plan isa before final r b=before r b) ∧
  (∀ r o, (r≠out.region ∨ ∀ x : Grid.Index g,o≠Grid.address g x) → final.mem r o=initial.mem r o) ∧
  (∀ x : Geometry.Cell g, ∃ pid,pid<Geometry.count g ∧ Geometry.Owns g pid x)

theorem native_complete (out : Host.Tensor) (t : TensorBinding.Tensor)
    (bound : TensorBinding.Binding out t) (g : Geometry.Layout) (valid : Geometry.Valid g)
    (shape : out.rows=g.rows ∧ out.columns=g.columns)
    (strides : out.rowStride=g.rowStride ∧ out.columnStride=g.columnStride)
    (width : Nat) (positive : 0<width) (half : Bool) (plan : Native.Plan) (isa : Native.WordISA)
    (before : Memory) (initial final : State Native.Expr) (reference : Grid.Index g → Native.Expr)
    (values : ∀ x : Grid.Index g,final.mem out.region (Grid.address g x)=floatCell half (reference x))
    (frame : ∀ r o,(r≠out.region ∨ ∀ pid,pid<Geometry.count g → ∀ i,
      Geometry.Active g pid i → o≠Geometry.offset g pid i) → final.mem r o=initial.mem r o) :
    NativeResult out t g width half plan isa before initial final reference := by
  refine ⟨values,?_,?_,serialize_frame out g width half plan isa before final,?_,Geometry.covers g valid⟩
  · funext r b
    simp only [serialize,NativeWhole.expected]
    split
    · split
      · rename_i hx
        rw [values (Classical.choose hx).1,float_read_cell]
      · rfl
    · rfl
  · intro x
    have hr : Geometry.outRow g x.1.val.1 x.1.val.2 < out.rows := by
      rw [shape.1];exact x.1.property.2.1
    have hn : Geometry.outColumn g x.1.val.1 x.1.val.2 < out.columns := by
      rw [shape.2];exact x.1.property.2.2
    rw [logicalIndex,KernelTensor.logical_byte out t bound width _ _ _ hr hn]
    have addr : width*(out.storageOffset+Geometry.outRow g x.1.val.1 x.1.val.2*out.rowStride+
        Geometry.outColumn g x.1.val.1 x.1.val.2*out.columnStride)+x.2.val=address out g width x := by
      simp only [address,Grid.address,Geometry.offset,strides.1,strides.2,Nat.add_assoc]
    rw [addr,serialize_read out g valid width positive half plan isa before final x,
      values x.1,float_read_cell]
  · intro r o outside
    apply frame r o
    exact outside.imp id (fun h pid hp i hi => h ⟨(pid,i),hp,hi⟩)

#specNonCircular RealResult avoiding [Grid.run]
#specNonCircular NativeResult avoiding [Grid.run]
#axiomsClean real_complete
#axiomsClean native_complete
end
end VeriTile.Bench.Repairs.MatmulDequantize.KernelResult
