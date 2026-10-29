import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperNativeAction
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperMaterialized

/-! Ordinary physical tensor descriptors and logical coordinate reads. These
contracts concern one tensor/view and contain no kernel-output equations. The
descriptor base already includes the tensor's storage offset. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 2000000

abbrev Descriptor := LayerNormFwd.Address.Allocation

def coordinateOffset : List Int → List Nat → Nat
  | stride::strides,index::indices => index*stride.toNat+coordinateOffset strides indices
  | _,_ => 0

def Coordinate (tensor : Tensor) (index : List Nat) : Prop :=
  List.Forall₂ (fun i n => i<n) index tensor.shape

/-- Nonnegative strides and signed metadata are ordinary Torch metadata
properties. Coordinate bounds account for views, gaps and storage offsets.
Logical negation is handled by the tensor API; raw reads are used only after
the already-proved successful `resolve_neg` result. -/
structure TensorMemory {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) : Prop where
  shape : ∀ s tensor, store.present s tensor → ∀ n ∈ tensor.shape, n<2^63
  strides : ∀ s tensor, store.present s tensor → tensor.strides.length=tensor.shape.length ∧
    ∀ stride ∈ tensor.strides, 0≤stride ∧ stride<2^63
  width : ∀ s tensor, store.present s tensor → (store.descriptor s tensor).width=tensor.dtype.bytes
  bounds : ∀ s tensor, store.present s tensor → ∀ index, Coordinate tensor index →
    coordinateOffset tensor.strides index<(store.descriptor s tensor).extent
  read : ∀ s tensor (hp : store.present s tensor), tensor.negative=Bool.false →
    ∀ index (hi : Coordinate tensor index),
      store.read s tensor index=s.memory
        ⟨LayerNormFwd.Address.byteAddress (store.descriptor s tensor) (coordinateOffset tensor.strides index),
          (LayerNormFwd.Address.byte_address (store.descriptor s tensor) (bounds s tensor hp index hi)).2.2.1⟩

/-- A materialized input is read directly from the current physical memory,
and this actual cell equals the original pre-wrapper logical coordinate. -/
theorem materialized_physical {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (original output : Tensor) (initial current : HostState α Host)
    (h : Materialized store original initial output current)
    (index : List Nat) (hi : Coordinate output index) :
    current.memory
      ⟨LayerNormFwd.Address.byteAddress (store.descriptor current output) (coordinateOffset output.strides index),
        (LayerNormFwd.Address.byte_address (store.descriptor current output)
          (hm.bounds current output h.present index hi)).2.2.1⟩=
      store.read initial original index := by
  rw [←hm.read current output h.present h.positive index hi]
  exact h.values index

theorem coordinate_matrix (tensor : Tensor) (M N row col : Nat)
    (hs : tensor.shape=[M,N]) (hr : row<M) (hc : col<N) : Coordinate tensor [row,col] := by
  unfold Coordinate
  rw [hs]
  exact .cons hr (.cons hc .nil)

theorem coordinate_vector (tensor : Tensor) (N index : Nat)
    (hs : tensor.shape=[N]) (hi : index<N) : Coordinate tensor [index] := by
  unfold Coordinate
  rw [hs]
  exact .cons hi .nil

theorem matrix_offset (stride : Int) (row col : Nat) :
    coordinateOffset [stride,1] [row,col]=row*stride.toNat+col := by
  simp [coordinateOffset]

theorem vector_offset (index : Nat) : coordinateOffset [1] [index]=index := by
  simp [coordinateOffset]

#axiomsClean materialized_physical
#axiomsClean coordinate_matrix
#axiomsClean coordinate_vector
#axiomsClean matrix_offset
#axiomsClean vector_offset
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
