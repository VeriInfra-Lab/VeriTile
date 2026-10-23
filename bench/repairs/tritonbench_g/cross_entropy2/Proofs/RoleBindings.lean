import bench.repairs.tritonbench_g.cross_entropy2.Proofs.BackwardReads
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.NativeEntry

namespace VeriTile.Bench.Repairs.CrossEntropy2.Physical
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations rowStride)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout Permissions Descriptor WritableDescriptor ReadAt WriteAt ScalarABI)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- Ordinary native descriptors for the exact tensors extracted from the
actual call. Labels are four-byte signed int32, including mixed aliases. -/
structure ForwardArguments {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {o : Options} {x y : Tensor}
    {rows cols : Nat} {initial final : World α σ} {returned : Return}
    {e : ForwardData M rt o x y rows cols initial returned final} (k : ForwardKernelData obs e) : Prop where
  scalar : ScalarABI e.parameters.toParameters e.count
  logits : Descriptor layout rights k.ready.native e.logits 4
  labels : Descriptor layout rights k.ready.native e.labels 4
  output : ∀ role, WritableDescriptor layout rights k.ready.native (k.output role) 4

structure BackwardArguments {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {o : Options} {g x l y : Tensor}
    {rows cols : Nat} {initial final : World α σ} {returned : Tensor}
    {e : BackwardData M rt o g x l y rows cols initial returned final} (k : BackwardKernelData obs e) : Prop where
  scalar : ScalarABI e.parameters.toParameters e.count
  logits : Descriptor layout rights k.ready.native e.logits 4
  labels : Descriptor layout rights k.ready.native e.labels 4
  upstream : Descriptor layout rights k.ready.native e.upstream 4
  lse : Descriptor layout rights k.ready.native e.lse 4
  gradient : WritableDescriptor layout rights k.ready.native returned 4

theorem packed_read {σ : Type} (layout : Layout σ) (rights : Permissions σ) (native : σ)
    (t : Tensor) (rows row : Nat) (descriptor : Descriptor layout rights native t 4)
    (shape : t.shape=[rows]) (packed : Packed t) (inside : row<rows) :
    ReadAt layout rights native t 4 (BitVec.ofNat 64 (t.storageOffset+row)) (t.storageOffset+row) := by
  have bound : row<t.numel := by simpa [Tensor.numel,shape] using inside
  have coordinate := packed.2 row bound
  have address := VeriTile.Bench.Repairs.CrossEntropy1.Physical.descriptor_address layout rights native t 4 descriptor (by decide) row bound
  refine ⟨by simpa only [coordinate] using address.1,?_⟩
  intro byte hb
  simpa only [coordinate] using descriptor.readable row byte bound hb

theorem forward_roles {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {o : Options} {x y : Tensor}
    {rows cols : Nat} {initial final : World α σ} {returned : Return}
    {e : ForwardData M rt o x y rows cols initial returned final} (k : ForwardKernelData obs e)
    (args : ForwardArguments layout rights k) (shape : x.shape=[rows,cols])
    (row block lane : Nat) (hr : row<rows) (hb : block<e.count) (hl : lane<o.width)
    (active : block*o.width+lane<cols) :
    ReadAt layout rights k.ready.native e.logits 4
      (VeriTile.Bench.Repairs.CrossEntropy1.Native.element e.parameters.logitsBase row e.parameters.logitsStride o.width block lane)
      (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress e.parameters.toParameters row (block*o.width+lane)) ∧
    ReadAt layout rights k.ready.native e.labels 4
      (BitVec.ofNat 64 (e.labels.storageOffset+row)) (e.labels.storageOffset+row) ∧
    ∀ role, WriteAt layout rights k.ready.native (k.output role) 4
      (VeriTile.Bench.Repairs.CrossEntropy1.Native.scalarOutput (k.output role).storageOffset rows row block)
      ((k.output role).storageOffset+block*rows+row) := by
  have logits := VeriTile.Bench.Repairs.CrossEntropy1.Physical.strided_address layout rights k.ready.native e.logits args.logits
    rows cols o.width e.count row block lane (k.logits.materialized.metadata.shape.trans shape) k.logits.rows args.scalar.grid hr hb hl active
  refine ⟨⟨logits.1,logits.2.2⟩,packed_read layout rights k.ready.native e.labels rows row args.labels
    (k.labels.materialized.metadata.shape.trans e.admitted.2.2.2.1) k.labels.packed hr,?_⟩
  intro role
  have dense : (k.output role).Dense := by
    cases role
    exacts [k.lossDense,k.lseDense,k.zDense]
  have stored := VeriTile.Bench.Repairs.CrossEntropy1.Physical.scalar_store_pointer layout rights k.ready.native (k.output role)
    e.count rows o.width cols block row (args.output role) args.scalar.grid dense (forward_arrays k role).shapeEq hb hr
  exact ⟨stored.1,stored.2⟩

/-- The target gather is guarded by ownership and block membership. Its
coordinate therefore lies in the actual logical input view. -/
theorem forward_target_role {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {o : Options} {x y : Tensor}
    {rows cols : Nat} {initial final : World α σ} {returned : Return}
    {e : ForwardData M rt o x y rows cols initial returned final} (k : ForwardKernelData obs e)
    (args : ForwardArguments layout rights k) (shape : x.shape=[rows,cols]) (row block : Nat) (hr : row<rows)
    (target : Int) (selected : blockOwns e.parameters target block) :
    let i := VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress e.parameters.toParameters row (target-o.start).toNat
    ReadAt layout rights k.ready.native e.logits 4 (BitVec.ofNat 64 i) i := by
  have col : (target-o.start).toNat<cols := by
    simp only [blockOwns,ForwardData.parameters,Host.parameters] at selected
    have nonnegative : (0 : Int) ≤ (block : Int)*(o.width : Int) :=
      mul_nonneg (Int.natCast_nonneg _) (Int.natCast_nonneg _)
    omega
  have shapeOut := k.logits.materialized.metadata.shape.trans shape
  have bound := VeriTile.Bench.Repairs.CrossEntropy1.Physical.row_coordinate_bound e.logits rows cols row _ shapeOut hr col
  have coordinate := k.logits.rows rows cols shapeOut row _ hr col
  have address := VeriTile.Bench.Repairs.CrossEntropy1.Physical.descriptor_address layout rights k.ready.native e.logits 4 args.logits (by decide) _ bound
  refine ⟨by simpa only [coordinate] using address.1,?_⟩
  intro byte hb
  simpa only [coordinate] using args.logits.readable _ byte bound hb

theorem backward_roles {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {o : Options} {g x l y : Tensor}
    {rows cols : Nat} {initial final : World α σ} {returned : Tensor}
    {e : BackwardData M rt o g x l y rows cols initial returned final} (k : BackwardKernelData obs e)
    (args : BackwardArguments layout rights k) (shape : x.shape=[rows,cols])
    (row block lane : Nat) (hr : row<rows) (hb : block<e.count) (hl : lane<o.width)
    (active : block*o.width+lane<cols) :
    ReadAt layout rights k.ready.native e.logits 4
      (VeriTile.Bench.Repairs.CrossEntropy1.Native.element e.parameters.logitsBase row e.parameters.logitsStride o.width block lane)
      (BackwardGrid.logitAddress e.parameters row (block*o.width+lane)) ∧
    ReadAt layout rights k.ready.native e.labels 4 (BitVec.ofNat 64 (e.labels.storageOffset+row)) (e.labels.storageOffset+row) ∧
    ReadAt layout rights k.ready.native e.lse 4 (BitVec.ofNat 64 (e.lse.storageOffset+row)) (e.lse.storageOffset+row) ∧
    ReadAt layout rights k.ready.native e.upstream 4
      (BitVec.ofNat 64 (e.upstream.storageOffset+row*rowStride e.upstream)) (e.upstream.storageOffset+row*rowStride e.upstream) ∧
    WriteAt layout rights k.ready.native returned 4
      (VeriTile.Bench.Repairs.CrossEntropy1.Native.element e.parameters.gradientBase row e.parameters.gradientStride o.width block lane)
      (BackwardGrid.gradientAddress e.parameters row (block*o.width+lane)) := by
  have logitsShape := k.logits.materialized.metadata.shape.trans shape
  have logits := VeriTile.Bench.Repairs.CrossEntropy1.Physical.strided_address layout rights k.ready.native e.logits args.logits
    rows cols o.width e.count row block lane logitsShape k.logits.rows args.scalar.grid hr hb hl active
  have upstream := VeriTile.Bench.Repairs.CrossEntropy1.Physical.vector_pointer layout rights k.ready.native e.upstream 4 rows row args.upstream
    (by decide) (k.upstream.materialized.metadata.shape.trans e.upstreamShape) k.upstream.materialized.rank hr
  have outShape := k.allocation.shapeEq.trans logitsShape
  have outLayout := VeriTile.Bench.Repairs.CrossEntropy1.Host.dense_row_layout returned k.dense k.allocation.rank
  have gradient := VeriTile.Bench.Repairs.CrossEntropy1.Physical.strided_address layout rights k.ready.native returned args.gradient.toDescriptor
    rows cols o.width e.count row block lane outShape outLayout args.scalar.grid hr hb hl active
  refine ⟨⟨logits.1,logits.2.2⟩,packed_read layout rights k.ready.native e.labels rows row args.labels
    (k.labels.materialized.metadata.shape.trans e.admitted.2.2.2.1) k.labels.packed hr,
    packed_read layout rights k.ready.native e.lse rows row args.lse
      (k.lse.materialized.metadata.shape.trans e.lseShape) k.lse.packed hr,
    ⟨upstream.1,upstream.2⟩,⟨gradient.1,?_⟩⟩
  have bound := VeriTile.Bench.Repairs.CrossEntropy1.Physical.row_coordinate_bound returned rows cols row _ outShape hr active
  have coordinate := outLayout rows cols outShape row _ hr active
  intro byte hbyte
  simpa only [coordinate] using args.gradient.writable _ byte bound hbyte

#axiomsClean forward_roles
#axiomsClean forward_target_role
#axiomsClean backward_roles
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Physical
