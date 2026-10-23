import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ForwardViews
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.BackwardKernel
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.RoleBindings

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Physical
open VeriTile.Triton
open Host
open CrossEntropy1.Source (FP32)
open CrossEntropy1.Host (World Observations ObjectRegistry RowLayout)
open CrossEntropy1.Physical (Layout Permissions Descriptor WritableDescriptor strided_address row_coordinate_bound
  dense_read_pointer scalar_store_pointer vector_pointer descriptor_address target_coordinate)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
namespace Native
abbrev Supported := CrossEntropy1.Native.Supported
abbrev address := CrossEntropy1.Native.address
abbrev element := CrossEntropy1.Native.element
abbrev scalarOutput := CrossEntropy1.Native.scalarOutput
end Native
noncomputable section
set_option maxHeartbeats 8000000

/-- Actual successful signed-i64 binder and launch metadata. The unannotated
ignore scalar is deliberately not restricted to signed-i64: its u64 branch
is covered separately. No extra host tensor-size cap is imposed. -/
structure ScalarABI (p : Parameters) (count : Nat) : Prop where
  grid : Native.Supported p.rows count p.block p.cols
  total : p.totalClasses<2^63
  startLow : -(2^63 : Int)≤p.classStart
  startHigh : p.classStart<2^63
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
  zloss : WritableDescriptor layout rights k.launch.ready.native f.zlosses 4

structure BackwardArguments {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {objects : ObjectRegistry rt.toRuntime} {c : Context}
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
    (active : CrossEntropy1.Source.column f.parameters.toParameters block lane<f.cols) :
    (Native.address (layout.buffer k.launch.ready.native f.workingLogits.region).base 4
      (Native.element f.parameters.logitsBase row f.parameters.logitsStride f.parameters.block block lane)).toNat=
      (layout.buffer k.launch.ready.native f.workingLogits.region).base+4*CrossEntropy1.Source.logitAddress f.parameters.toParameters row (CrossEntropy1.Source.column f.parameters.toParameters block lane) ∧
    ∀ byte, byte<4 → rights.readable k.launch.ready.native f.workingLogits.device
      ((layout.buffer k.launch.ready.native f.workingLogits.region).base+4*CrossEntropy1.Source.logitAddress f.parameters.toParameters row (CrossEntropy1.Source.column f.parameters.toParameters block lane)+byte) := by
  have access := strided_address layout rights k.launch.ready.native f.workingLogits args.logits
    f.rows f.cols f.parameters.block f.count row block lane (k.working.metadata.shape.trans f.logitsShape)
    k.layout args.scalar.grid hr hb hl active
  exact ⟨access.1,access.2.2⟩

theorem backward_store_pointer {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {objects : ObjectRegistry rt.toRuntime} {c : Context}
    {upstream output : Tensor} {initial final : World α σ}
    {b : BackwardData M rt c upstream output initial final} (k : BackwardKernelData obs objects b)
    (args : BackwardArguments layout rights k) (row block lane : Nat)
    (hr : row<b.rows) (hb : block<b.count) (hl : lane<b.parameters.block)
    (active : CrossEntropy1.Source.column b.parameters.toParameters block lane<b.cols) :
    (Native.address (layout.buffer k.launch.ready.native b.work.region).base 4
      (Native.element b.parameters.gradientBase row b.parameters.gradientStride b.parameters.block block lane)).toNat=
      (layout.buffer k.launch.ready.native b.work.region).base+4*CrossEntropy1.Source.gradientAddress b.parameters.toParameters row (CrossEntropy1.Source.column b.parameters.toParameters block lane) ∧
    ∀ byte, byte<4 → rights.writable k.launch.ready.native b.work.device
      ((layout.buffer k.launch.ready.native b.work.region).base+4*CrossEntropy1.Source.gradientAddress b.parameters.toParameters row (CrossEntropy1.Source.column b.parameters.toParameters block lane)+byte) := by
  have shape := k.work.buffer.shape.trans b.shape
  have access := strided_address layout rights k.launch.ready.native b.work args.gradient.toDescriptor
    b.rows b.cols b.parameters.block b.count row block lane shape k.work.layout args.scalar.grid hr hb hl active
  refine ⟨access.1,?_⟩
  have bound := row_coordinate_bound b.work b.rows b.cols row (CrossEntropy1.Source.column b.parameters.toParameters block lane) shape hr active
  have coordinate := k.work.layout b.rows b.cols shape row (CrossEntropy1.Source.column b.parameters.toParameters block lane) hr active
  intro byte hbyte
  simpa only [coordinate] using args.gradient.writable _ byte bound hbyte

theorem forward_scalar_roles {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {f : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs f)
    (args : ForwardArguments layout rights k) (row block : Nat) (hr : row<f.rows) (hb : block<f.count) :
    (Native.address (layout.buffer k.launch.ready.native f.labels.region).base 8
      (BitVec.ofNat 64 (f.parameters.labelsBase+row))).toNat=
      (layout.buffer k.launch.ready.native f.labels.region).base+8*(f.parameters.labelsBase+row) ∧
    (∀ byte, byte<8 → rights.readable k.launch.ready.native f.labels.device
      ((layout.buffer k.launch.ready.native f.labels.region).base+8*(f.parameters.labelsBase+row)+byte)) ∧
    (∀ t, t=f.losses ∨ t=f.lse ∨ t=f.zlosses →
      (Native.address (layout.buffer k.launch.ready.native t.region).base 4
        (Native.scalarOutput t.storageOffset f.rows row block)).toNat=
        (layout.buffer k.launch.ready.native t.region).base+4*(t.storageOffset+block*f.rows+row) ∧
      ∀ byte, byte<4 → rights.writable k.launch.ready.native t.device
        ((layout.buffer k.launch.ready.native t.region).base+4*(t.storageOffset+block*f.rows+row)+byte)) := by
  have labels := dense_read_pointer layout rights k.launch.ready.native f.labels 8 f.rows row args.labels (by decide)
    (k.labelsSnapshot.materialized.metadata.shape.trans f.labelsShape) k.labelsSnapshot.dense hr
  refine ⟨labels.1,labels.2,?_⟩
  intro t choice
  rcases choice with rfl|rfl|rfl
  · exact scalar_store_pointer layout rights k.launch.ready.native f.losses f.count f.rows f.parameters.block f.cols block row
      args.loss args.scalar.grid k.lossDense k.lossAllocation.shapeEq hb hr
  · exact scalar_store_pointer layout rights k.launch.ready.native f.lse f.count f.rows f.parameters.block f.cols block row
      args.lse args.scalar.grid k.lseDense k.lseAllocation.shapeEq hb hr
  · exact scalar_store_pointer layout rights k.launch.ready.native f.zlosses f.count f.rows f.parameters.block f.cols block row
      args.zloss args.scalar.grid k.zDense k.zAllocation.shapeEq hb hr

theorem forward_target_pointer {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {f : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs f)
    (args : ForwardArguments layout rights k) (row block : Nat) (hr : row<f.rows)
    (target : Int) (branch : (block*f.parameters.block : Nat)≤target-f.parameters.classStart ∧
      target-f.parameters.classStart<(min f.cols ((block+1)*f.parameters.block) : Nat)) :
    let col := (target-f.parameters.classStart).toNat
    (Native.address (layout.buffer k.launch.ready.native f.workingLogits.region).base 4
      (BitVec.ofNat 64 (CrossEntropy1.Source.logitAddress f.parameters.toParameters row col))).toNat=
      (layout.buffer k.launch.ready.native f.workingLogits.region).base+4*CrossEntropy1.Source.logitAddress f.parameters.toParameters row col ∧
    ∀ byte, byte<4 → rights.readable k.launch.ready.native f.workingLogits.device
      ((layout.buffer k.launch.ready.native f.workingLogits.region).base+4*CrossEntropy1.Source.logitAddress f.parameters.toParameters row col+byte) := by
  have col := (target_coordinate target f.parameters.classStart f.cols f.parameters.block block branch).2
  have shape := k.working.metadata.shape.trans f.logitsShape
  have bound := row_coordinate_bound f.workingLogits f.rows f.cols row _ shape hr col
  have coordinate := k.layout f.rows f.cols shape row _ hr col
  have address := descriptor_address layout rights k.launch.ready.native f.workingLogits 4 args.logits (by decide) _ bound
  exact ⟨by simpa only [coordinate] using address.1,
    fun byte hbyte => by simpa only [coordinate] using args.logits.readable _ byte bound hbyte⟩

#axiomsClean forward_logit_pointer
#axiomsClean forward_scalar_roles
#axiomsClean forward_target_pointer
#axiomsClean backward_store_pointer
#specNonCircular ScalarABI avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Physical
