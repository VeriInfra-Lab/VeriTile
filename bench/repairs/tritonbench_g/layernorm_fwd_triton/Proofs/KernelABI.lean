import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.KernelInputs
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.KernelABI
open VeriTile.Triton FP.Structural Address InputBinding
noncomputable section

/-- The actual widened products and pointer updates execute modulo 64 bits,
including unused row bases, masked lanes, and the final loop increment. -/
def indexWord (s h col ss sh sd : Nat) : BitVec 64 :=
  BitVec.ofNat 64 s*BitVec.ofNat 64 ss+
    BitVec.ofNat 64 h*BitVec.ofNat 64 sh+BitVec.ofNat 64 col*BitVec.ofNat 64 sd

def pointer (p : BitVec 64) (s h col ss sh sd : Nat) : BitVec 64 :=
  p+indexWord s h col ss sh sd*BitVec.ofNat 64 4

theorem index_mod (s h col ss sh sd : Nat) :
    indexWord s h col ss sh sd=BitVec.ofNat 64 (offset s h col ss sh sd) := by
  simp only [indexWord,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,offset]

theorem pointer_mod (p : BitVec 64) (s h col ss sh sd : Nat) :
    pointer p s h col ss sh sd=p+BitVec.ofNat 64 (4*offset s h col ss sh sd) := by
  simp [pointer,index_mod,BitVec.ofNat_mul_ofNat,Nat.mul_comm]

theorem increment (base : BitVec 64) (off stride : Nat) :
    (base+BitVec.ofNat 64 (4*(off*stride)))+BitVec.ofNat 64 (4*(128*stride))=
      base+BitVec.ofNat 64 (4*((off+128)*stride)) := by
  rw [BitVec.add_assoc,BitVec.ofNat_add_ofNat]
  congr 2
  ring

variable {shape : Shape} {st : Strides} {eps : Contract.Epsilon}

theorem active_pointers (a : Arguments shape st eps) (row : Row shape) (col : Fin shape.columns) :
    (pointer a.x.dataPtr row.1.val row.2.val col.val st.xSeq st.xHead st.xDim).toNat=
      address a.x (xOffset st row col.val) ∧
    (pointer a.w.dataPtr 0 row.2.val col.val 0 st.wHead st.wDim).toNat=
      address a.w (wOffset st row col.val) ∧
    (pointer a.y.dataPtr row.1.val row.2.val col.val st.ySeq st.yHead st.yDim).toNat=
      address a.y (yOffset st row col.val) := by
  simp only [pointer_mod]
  exact ⟨(x_pointer a (fun _ => FP.Value.zero .fp32) row col).2.2.1,
    (w_pointer a (fun _ => FP.Value.zero .fp32) row col).2.2.1,(y_pointer a row col).2.2⟩

/-- Ordinary output capacity bounds the executed loop even for broadcast
inputs. No positive count or artificial small-shape guard is added. -/
theorem loop_bound (a : Arguments shape st eps) (row : Row shape) : shape.columns+128<2^63 := by
  let embed : Fin shape.columns → Fin a.y.storage.elements := fun col =>
    ⟨a.y.physicalIndex (index3 row col.val),a.yLaws.inBounds _ (by
      simpa [VeriTile.Torch.TensorMetadata.numel,a.yView.dimensions] using index3_lt row col)⟩
  have inj : Function.Injective embed := by
    intro c d he
    have hv := congrArg Fin.val he
    change a.y.physicalIndex (index3 row c.val)=a.y.physicalIndex (index3 row d.val) at hv
    rw [a.yView.index row c,a.yView.index row d] at hv
    have ho := Nat.add_left_cancel hv
    have hq := a.outputInjective (a₁:=(row,c)) (a₂:=(row,d)) ho
    exact congrArg Prod.snd hq
  have hn : shape.columns≤a.y.storage.elements := by
    simpa using Fintype.card_le_of_injective embed inj
  have span := a.y.storage.spanFits
  omega

theorem loop_exact (a : Arguments shape st eps) (row : Row shape) (off : Nat)
    (active : off<shape.columns) (lane : Fin 128) :
    VeriTile.Bench.Repairs.LayerNormFwd.Address.signed64 (off+lane.val)=(off+lane.val:Nat) ∧
    VeriTile.Bench.Repairs.LayerNormFwd.Address.signed64 (off+128)=(off+128:Nat) := by
  have hb := loop_bound a row
  exact ⟨VeriTile.Bench.Repairs.LayerNormFwd.Address.signed64_exact (by omega),
    VeriTile.Bench.Repairs.LayerNormFwd.Address.signed64_exact (by omega)⟩

theorem active_alignment (a : Arguments shape st eps) (row : Row shape) (col : Fin shape.columns) :
    address a.x (xOffset st row col.val)%4=0 ∧
    address a.w (wOffset st row col.val)%4=0 ∧
    address a.y (yOffset st row col.val)%4=0 := by
  have hx : 0<a.x.metadata.numel := by
    have h := index3_lt row col
    simp only [VeriTile.Torch.TensorMetadata.numel,a.xView.dimensions,List.prod_cons,List.prod_nil,Nat.mul_one]
    omega
  have hw : 0<a.w.metadata.numel := by
    have h := index2_lt row col
    simp only [VeriTile.Torch.TensorMetadata.numel,a.wView.dimensions,List.prod_cons,List.prod_nil,Nat.mul_one]
    omega
  have hy : 0<a.y.metadata.numel := by
    have h := index3_lt row col
    simp only [VeriTile.Torch.TensorMetadata.numel,a.yView.dimensions,List.prod_cons,List.prod_nil,Nat.mul_one]
    omega
  have ax : a.x.dataPtr.toNat%4=0 := a.admitted.input.alignment hx
  have aw : a.w.dataPtr.toNat%4=0 := a.admitted.weight.alignment hw
  simp [address,Nat.add_mod,ax,aw,a.yAligned hy]

/-- Scalar packing retains its actual selected kind; an admitted Python int,
float or bool need not have an FP32 wire representation. -/
inductive HostScalar where
  | integer (value : Int) | binary64 (bits : BitVec 64) | boolean (value : Bool)
inductive ScalarArgument where
  | signed32 (bits : BitVec 32) | unsigned32 (bits : BitVec 32)
  | signed64 (bits : BitVec 64) | unsigned64 (bits : BitVec 64)
  | fp32 (bits : BitVec 32) | fp64 (bits : BitVec 64)
  | boolean (value : Bool) | specialized (value : Int)

/-- Local ABI interpretation ports, never an execution/reduction/result port.
Host classification, packing and conversion semantics are modeling boundaries. -/
structure ScalarSemantics where
  hostValue : HostScalar → ℝ
  packed : HostScalar → ScalarArgument → Prop
  convert32 : ScalarArgument → NativeKernel.Word

structure EpsilonBinding (eps : Contract.Epsilon) where
  semantics : ScalarSemantics
  host : HostScalar
  argument : ScalarArgument
  host_value : semantics.hostValue host=eps.value
  packed : semantics.packed host argument

/-- The actual ABI conversion is consumed by the source register and graph. -/
def EpsilonBinding.term (abi : EpsilonBinding eps) : NativeKernel.Term :=
  NativeKernel.epsilonTerm (abi.semantics.convert32 abi.argument)

theorem EpsilonBinding.observation (abi : EpsilonBinding eps)
    (I : NativeKernel.Instructions) (P : NativeKernel.Profile) :
    NativeKernel.observe I P abi.term=abi.semantics.convert32 abi.argument := rfl

#axiomsClean index_mod
#axiomsClean pointer_mod
#axiomsClean increment
#axiomsClean active_pointers
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.KernelABI
