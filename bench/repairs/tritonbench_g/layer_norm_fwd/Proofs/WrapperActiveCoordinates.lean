import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperRoleCoordinates

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

def TensorCoordinate {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (s : HostState α Host)
    (tensor : Tensor) (index : List Nat) (offset : Nat) : Prop :=
  store.present s tensor ∧ tensor.negative=Bool.false ∧ Coordinate tensor index ∧
    offset=coordinateOffset tensor.strides index

theorem tensor_matrix_coordinate {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (s : HostState α Host) (tensor : Tensor) (hp : store.present s tensor) (hn : tensor.negative=Bool.false)
    (M N : Nat) (hs : tensor.shape=[M,N]) (stride : Int) (hf : tensor.strides[0]?=some stride)
    (hd : InnerDense tensor) (row col : Nat) (hr : row<M) (hc : col<N) :
    TensorCoordinate store s tensor [row,col] (row*stride.toNat+col) :=
  ⟨hp,hn,coordinate_matrix tensor M N row col hs hr hc,
    (matrix_active_offset tensor M N stride hs (hm.strides s tensor hp).1 hf hd row col hc).symm⟩

theorem tensor_vector_coordinate {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (s : HostState α Host) (tensor : Tensor) (hp : store.present s tensor) (hn : tensor.negative=Bool.false)
    (N : Nat) (hs : tensor.shape=[N]) (hd : InnerDense tensor) (index : Nat) (hi : index<N) :
    TensorCoordinate store s tensor [index] index :=
  ⟨hp,hn,coordinate_vector tensor N index hs hi,
    (vector_active_offset tensor N hs (hm.strides s tensor hp).1 hd index hi).symm⟩

theorem tensor_dense_vector_coordinate {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (s : HostState α Host) (tensor : Tensor) (hp : store.present s tensor) (hn : tensor.negative=Bool.false)
    (N : Nat) (hs : tensor.shape=[N]) (hd : ContiguousLayout tensor) (index : Nat) (hi : index<N) :
    TensorCoordinate store s tensor [index] index :=
  ⟨hp,hn,coordinate_vector tensor N index hs hi,(contiguous_vector_offset tensor N hs hd index hi).symm⟩

theorem tensor_dense_matrix_coordinate {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (s : HostState α Host) (tensor : Tensor) (hp : store.present s tensor) (hn : tensor.negative=Bool.false)
    (M N : Nat) (hs : tensor.shape=[M,N]) (hd : ContiguousLayout tensor)
    (row col : Nat) (hr : row<M) (hc : col<N) :
    TensorCoordinate store s tensor [row,col] (row*N+col) :=
  ⟨hp,hn,coordinate_matrix tensor M N row col hs hr hc,
    (contiguous_matrix_offset tensor M N hs hd row col hr hc).symm⟩

/-- All actual scalar and vector accesses are valid tensor coordinates.
Scalar roles do not acquire a col<N premise, so N=0 keeps its genuine scalar
loads/stores. The impossible x1+rowscale path follows from source validation. -/
theorem active_role_coordinate {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (row col : Nat) (hr : row<p.validated.M) (role : LayerNormFwd.Address.Role)
    (ha : LayerNormFwd.Address.active (p.flags q) role)
    (hl : NativeAccess.laneActive p.validated.N col role) :
    RoleCoordinate store current p args row col role := by
  have hxshape := h.inputs.x.shape.trans hv.x.1
  have hyshape : p.outputs.y.shape=[p.validated.M,p.validated.N] := h.outputs.1.shape.trans hxshape
  cases role with
  | x =>
    exact ⟨p.inputs.x,rfl,tensor_matrix_coordinate store hm current p.inputs.x h.inputs.x.present
      h.inputs.x.positive _ _ hxshape _ h.arguments.x hi.x row col hr hl⟩
  | y =>
    exact ⟨p.outputs.y,rfl,tensor_matrix_coordinate store hm current p.outputs.y h.outputs.1.present
      h.outputs.1.positive _ _ hyshape _ h.arguments.y ho.y row col hr hl⟩
  | weight =>
    exact ⟨p.inputs.weight,rfl,tensor_vector_coordinate store hm current p.inputs.weight h.inputs.weight.present
      h.inputs.weight.positive _ (h.inputs.weight.shape.trans hv.weight.1) hi.weight col hl⟩
  | residual =>
    obtain ⟨out,he⟩ := tensor_of_isSome p.inputs.residual ha
    obtain ⟨original,horg,hmade⟩ := materialized_of_some h.inputs.residual out he
    have hstride : out.strides[0]?=some args.strideResidual := by
      simpa only [OptionalStrideBound,he] using h.arguments.residual
    exact ⟨out,he,tensor_matrix_coordinate store hm current out hmade.present hmade.positive _ _
      (hmade.shape.trans (hv.residual original horg).1) _ hstride (hi.residual out he) row col hr hl⟩
  | x1 =>
    obtain ⟨out,he⟩ := tensor_of_isSome p.inputs.x1 ha
    obtain ⟨original,horg,hmade⟩ := materialized_of_some h.inputs.x1 out he
    have hstride : out.strides[0]?=some args.strideX1 := by
      simpa only [OptionalStrideBound,he] using h.arguments.x1
    exact ⟨out,he,tensor_matrix_coordinate store hm current out hmade.present hmade.positive _ _
      (hmade.shape.trans ((hv.x1 original horg).1.trans hv.x.1)) _ hstride (hi.x1 out he) row col hr hl⟩
  | bias =>
    obtain ⟨out,he⟩ := tensor_of_isSome p.inputs.bias ha
    obtain ⟨original,horg,hmade⟩ := materialized_of_some h.inputs.bias out he
    exact ⟨out,he,tensor_vector_coordinate store hm current out hmade.present hmade.positive _
      (hmade.shape.trans (hv.bias original horg).1) (hi.bias out he) col hl⟩
  | weight1 =>
    obtain ⟨out,he⟩ := tensor_of_isSome p.inputs.weight1 ha
    obtain ⟨original,horg,hmade⟩ := materialized_of_some h.inputs.weight1 out he
    exact ⟨out,he,tensor_vector_coordinate store hm current out hmade.present hmade.positive _
      (hmade.shape.trans (hv.weight1 original horg).1) (hi.weight1 out he) col hl⟩
  | bias1 =>
    obtain ⟨out,he⟩ := tensor_of_isSome p.inputs.bias1 ha.2
    obtain ⟨original,horg,hmade⟩ := materialized_of_some h.inputs.bias1 out he
    exact ⟨out,he,tensor_vector_coordinate store hm current out hmade.present hmade.positive _
      (hmade.shape.trans (hv.bias1 original horg).1) (hi.bias1 out he) col hl⟩
  | y1 =>
    obtain ⟨out,he,halloc⟩ := allocated_of_enabled h.outputs.2.1 ha
    have hstride : out.strides[0]?=some args.strideY1 := by
      simpa only [OptionalStrideBound,he] using h.arguments.y1
    exact ⟨out,he,tensor_matrix_coordinate store hm current out halloc.present halloc.positive _ _
      (halloc.shape.trans hyshape) _ hstride (ho.y1 out he) row col hr hl⟩
  | residualOut =>
    obtain ⟨out,he⟩ := tensor_of_isSome p.outputs.residualOut ha
    have halloc := allocated_of_some h.outputs.2.2.1 out he
    have hstride : out.strides[0]?=some args.strideResidualOut := by
      simpa only [OptionalStrideBound,he] using h.arguments.residualOut
    exact ⟨out,he,tensor_matrix_coordinate store hm current out halloc.present halloc.positive _ _
      halloc.shape _ hstride (ho.residualOut out he) row col hr hl⟩
  | rowscale =>
    obtain ⟨out,he⟩ := tensor_of_isSome p.inputs.rowscale ha
    obtain ⟨original,horg,hmade⟩ := materialized_of_some h.inputs.rowscale out he
    exact ⟨out,he,tensor_dense_vector_coordinate store current out hmade.present hmade.positive _
      (hmade.shape.trans (hv.rowscale original horg).1) (hi.rowscale out he) row hr⟩
  | rowscale1 => exact False.elim (prepared_no_second_rowscale store tensor q initial current p args h hv ha)
  | mean =>
    have henabled : (!q.rms)=Bool.true := by simpa [LayerNormFwd.Address.active,Prepared.flags] using ha
    obtain ⟨out,he,halloc⟩ := allocated_of_enabled h.outputs.2.2.2.1 henabled
    exact ⟨out,he,tensor_dense_vector_coordinate store current out halloc.present halloc.positive _
      halloc.shape (ho.mean out he) row hr⟩
  | rstd =>
    exact ⟨p.outputs.rstd,rfl,tensor_dense_vector_coordinate store current p.outputs.rstd
      h.outputs.2.2.2.2.1.present h.outputs.2.2.2.2.1.positive _ h.outputs.2.2.2.2.1.shape ho.rstd row hr⟩
  | seeds =>
    obtain ⟨out,he,halloc⟩ := allocated_of_enabled h.outputs.2.2.2.2.2.1 ha
    have hrow : row<(if p.inputs.x1.isSome then 2*p.validated.M else p.validated.M) := by
      split <;> omega
    exact ⟨out,he,tensor_dense_vector_coordinate store current out halloc.present halloc.positive _
      halloc.shape (ho.seeds out he) row hrow⟩
  | seeds1 =>
    obtain ⟨out,he,halloc⟩ := allocated_of_enabled h.outputs.2.2.2.2.2.1 ha.2
    have hx : p.inputs.x1.isSome=Bool.true := ha.1
    have hshape : out.shape=[2*p.validated.M] := by simpa [outputRequests,hx] using halloc.shape
    exact ⟨out,he,tensor_dense_vector_coordinate store current out halloc.present halloc.positive _
      hshape (ho.seeds out he) (p.validated.M+row) (by omega)⟩
  | mask =>
    obtain ⟨out,he⟩ := tensor_of_isSome p.outputs.mask ha.2
    have halloc := allocated_of_some h.outputs.2.2.2.2.2.2 out he
    have hrow : row<(if p.inputs.x1.isSome then 2*p.validated.M else p.validated.M) := by
      split <;> omega
    exact ⟨out,he,tensor_dense_matrix_coordinate store current out halloc.present halloc.positive _ _
      halloc.shape (ho.mask out he) row col hrow hl⟩
  | mask1 =>
    obtain ⟨out,he⟩ := tensor_of_isSome p.outputs.mask ha.2.2
    have halloc := allocated_of_some h.outputs.2.2.2.2.2.2 out he
    have hx : p.inputs.x1.isSome=Bool.true := ha.1
    have hshape : out.shape=[2*p.validated.M,p.validated.N] := by simpa [outputRequests,hx] using halloc.shape
    exact ⟨out,he,tensor_dense_matrix_coordinate store current out halloc.present halloc.positive _ _
      hshape (ho.mask out he) (p.validated.M+row) col (by omega) hl⟩

#axiomsClean tensor_matrix_coordinate
#axiomsClean tensor_vector_coordinate
#axiomsClean tensor_dense_vector_coordinate
#axiomsClean tensor_dense_matrix_coordinate
#axiomsClean active_role_coordinate
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
