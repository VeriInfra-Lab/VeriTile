import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Descriptors
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.BackwardReads

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

/-- Local launcher ABI. Annotated int64 scalar conversion and CUDA dimension
limits are ordinary setup contracts; no output or access formula is assumed. -/
structure ScalarABI (p : Parameters) (count : Nat) : Prop where
  grid : Native.Supported p.rows count p.block p.cols
  total : p.totalClasses<2^63
  startLow : -(2^63 : Int)≤p.classStart
  startHigh : p.classStart<2^63
  ignoredLow : -(2^63 : Int)≤p.ignored
  ignoredHigh : p.ignored<2^63
  logitsStride : p.logitsStride<2^63
  gradientStride : p.gradientStride<2^63
  upstreamStride : p.upstreamStride<2^63

/-- Actual forward tensor arguments, described at the call boundary. Each
backing condition concerns the tensor's own logical coordinates; the source
index arithmetic is discharged by the following theorems. -/
structure ForwardArguments {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {f : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs f) : Prop where
  scalar : ScalarABI f.parameters f.count
  logits : Descriptor layout rights k.launch.ready.native f.workingLogits 4
  labels : Descriptor layout rights k.launch.ready.native f.labels 8
  loss : WritableDescriptor layout rights k.launch.ready.native f.losses 4
  lse : WritableDescriptor layout rights k.launch.ready.native f.lse 4

structure BackwardArguments {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {objects : ObjectRegistry rt} {c : Context}
    {upstream output : Tensor} {initial final : World α σ}
    {b : BackwardData M rt c upstream output initial final} (k : BackwardKernelData obs objects b) : Prop where
  scalar : ScalarABI b.parameters b.count
  logits : Descriptor layout rights k.launch.ready.native c.logits 4
  labels : Descriptor layout rights k.launch.ready.native c.labels 8
  lse : Descriptor layout rights k.launch.ready.native c.lse 4
  upstream : Descriptor layout rights k.launch.ready.native b.upstream 4
  gradient : WritableDescriptor layout rights k.launch.ready.native b.work 4

theorem forward_logit_pointer {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {f : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs f)
    (args : ForwardArguments layout rights k) (row block lane : Nat)
    (hr : row<f.rows) (hb : block<f.count) (hl : lane<f.parameters.block)
    (active : column f.parameters block lane<f.cols) :
    (Native.address (layout.buffer k.launch.ready.native f.workingLogits.region).base 4
      (Native.element f.parameters.logitsBase row f.parameters.logitsStride f.parameters.block block lane)).toNat=
      (layout.buffer k.launch.ready.native f.workingLogits.region).base+4*logitAddress f.parameters row (column f.parameters block lane) ∧
    ∀ byte, byte<4 → rights.readable k.launch.ready.native f.workingLogits.device
      ((layout.buffer k.launch.ready.native f.workingLogits.region).base+4*logitAddress f.parameters row (column f.parameters block lane)+byte) := by
  have access := strided_address layout rights k.launch.ready.native f.workingLogits args.logits
    f.rows f.cols f.parameters.block f.count row block lane (k.working.metadata.shape.trans f.logitsShape)
    k.layout args.scalar.grid hr hb hl active
  exact ⟨access.1,access.2.2⟩

theorem backward_store_pointer {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {objects : ObjectRegistry rt} {c : Context}
    {upstream output : Tensor} {initial final : World α σ}
    {b : BackwardData M rt c upstream output initial final} (k : BackwardKernelData obs objects b)
    (args : BackwardArguments layout rights k) (row block lane : Nat)
    (hr : row<b.rows) (hb : block<b.count) (hl : lane<b.parameters.block)
    (active : column b.parameters block lane<b.cols) :
    (Native.address (layout.buffer k.launch.ready.native b.work.region).base 4
      (Native.element b.parameters.gradientBase row b.parameters.gradientStride b.parameters.block block lane)).toNat=
      (layout.buffer k.launch.ready.native b.work.region).base+4*gradientAddress b.parameters row (column b.parameters block lane) ∧
    ∀ byte, byte<4 → rights.writable k.launch.ready.native b.work.device
      ((layout.buffer k.launch.ready.native b.work.region).base+4*gradientAddress b.parameters row (column b.parameters block lane)+byte) := by
  have shape := k.work.buffer.shape.trans b.shape
  have access := strided_address layout rights k.launch.ready.native b.work args.gradient.toDescriptor
    b.rows b.cols b.parameters.block b.count row block lane shape k.work.layout args.scalar.grid hr hb hl active
  refine ⟨access.1,?_⟩
  have bound := row_coordinate_bound b.work b.rows b.cols row (column b.parameters block lane) shape hr active
  have coordinate := k.work.layout b.rows b.cols shape row (column b.parameters block lane) hr active
  intro byte hbyte
  simpa only [coordinate] using args.gradient.writable _ byte bound hbyte

theorem vector_pointer {σ : Type} (layout : Layout σ) (rights : Permissions σ) (native : σ)
    (t : Tensor) (width rows row : Nat) (descriptor : Descriptor layout rights native t width)
    (positive : 0<width) (shape : t.shape=[rows]) (rank : t.strides.length=t.shape.length) (inside : row<rows) :
    (Native.address (layout.buffer native t.region).base width
      (BitVec.ofNat 64 (t.storageOffset+row*rowStride t))).toNat=
      (layout.buffer native t.region).base+width*(t.storageOffset+row*rowStride t) ∧
    ∀ byte, byte<width → rights.readable native t.device
      ((layout.buffer native t.region).base+width*(t.storageOffset+row*rowStride t)+byte) := by
  have bound : row<t.numel := by simpa [Tensor.numel,shape] using inside
  have coordinate := vector_layout t rows row shape rank inside
  have address := descriptor_address layout rights native t width descriptor positive row bound
  refine ⟨by simpa only [coordinate] using address.1,?_⟩
  intro byte hbyte
  simpa only [coordinate] using descriptor.readable row byte bound hbyte

theorem scalar_output_coordinate (t : Tensor) (count rows block row : Nat)
    (dense : t.Dense) (shape : t.shape=if count>1 then [count,rows] else [rows])
    (hb : block<count) (hr : row<rows) :
    block*rows+row<t.numel ∧ t.index (block*rows+row)=t.storageOffset+block*rows+row := by
  have bound : block*rows+row<t.numel := by
    by_cases many : count>1
    · simp only [Tensor.numel,shape,if_pos many,List.prod_cons,List.prod_nil,Nat.mul_one]
      have product := Nat.mul_le_mul_right rows (Nat.succ_le_of_lt hb)
      simp only [Nat.succ_mul] at product
      omega
    · have blockZero : block=0 := by omega
      simp only [Tensor.numel,shape,if_neg many,List.prod_cons,List.prod_nil,Nat.mul_one,blockZero,Nat.zero_mul,Nat.zero_add]
      exact hr
  have base := dense_base t dense (by omega)
  exact ⟨bound,by rw [dense.2 _ bound,base,Nat.zero_add]⟩

theorem scalar_store_pointer {σ : Type} (layout : Layout σ) (rights : Permissions σ) (native : σ)
    (t : Tensor) (count rows width cols block row : Nat) (descriptor : WritableDescriptor layout rights native t 4)
    (api : Native.Supported rows count width cols) (dense : t.Dense)
    (shape : t.shape=if count>1 then [count,rows] else [rows]) (hb : block<count) (hr : row<rows) :
    (Native.address (layout.buffer native t.region).base 4 (Native.scalarOutput t.storageOffset rows row block)).toNat=
      (layout.buffer native t.region).base+4*(t.storageOffset+block*rows+row) ∧
    ∀ byte, byte<4 → rights.writable native t.device
      ((layout.buffer native t.region).base+4*(t.storageOffset+block*rows+row)+byte) := by
  have coordinate := scalar_output_coordinate t count rows block row dense shape hb hr
  have address := descriptor_address layout rights native t 4 descriptor.toDescriptor (by decide) _ coordinate.1
  refine ⟨?_,?_⟩
  · rw [Native.scalar_output_word t.storageOffset rows row block count width cols api hr hb]
    simpa only [coordinate.2] using address.1
  · intro byte hbyte
    simpa only [coordinate.2] using descriptor.writable _ byte coordinate.1 hbyte

#axiomsClean forward_logit_pointer
#axiomsClean backward_store_pointer
#axiomsClean vector_pointer
#axiomsClean scalar_store_pointer
#specNonCircular ScalarABI avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
