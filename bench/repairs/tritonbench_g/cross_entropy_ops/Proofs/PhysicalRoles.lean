import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.PhysicalBindings
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Entries

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Physical
open VeriTile.Triton Host
open CrossEntropy1.Source (FP32)
open CrossEntropy1.Host (World Observations ObjectRegistry RowLayout)
open CrossEntropy1.Physical (Layout Permissions Descriptor WritableDescriptor strided_address vector_pointer dense_read_pointer)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

theorem backward_logit_pointer {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {objects : ObjectRegistry rt.toRuntime} {c : Context}
    {upstream output : Tensor} {initial final : World α σ}
    {b : BackwardData M rt c upstream output initial final} (k : BackwardKernelData obs objects b)
    (args : BackwardArguments layout rights k) (geometry : RowLayout c.logits) (row block lane : Nat)
    (hr : row<b.rows) (hb : block<b.count) (hl : lane<b.parameters.block)
    (active : CrossEntropy1.Source.column b.parameters.toParameters block lane<b.cols) :
    (Native.address (layout.buffer k.launch.ready.native c.logits.region).base 4
      (Native.element b.parameters.logitsBase row b.parameters.logitsStride b.parameters.block block lane)).toNat=
      (layout.buffer k.launch.ready.native c.logits.region).base+4*CrossEntropy1.Source.logitAddress b.parameters.toParameters row (CrossEntropy1.Source.column b.parameters.toParameters block lane) ∧
    ∀ byte, byte<4 → rights.readable k.launch.ready.native c.logits.device
      ((layout.buffer k.launch.ready.native c.logits.region).base+4*CrossEntropy1.Source.logitAddress b.parameters.toParameters row (CrossEntropy1.Source.column b.parameters.toParameters block lane)+byte) := by
  have pointer := strided_address layout rights k.launch.ready.native c.logits args.logits
    b.rows b.cols b.parameters.block b.count row block lane b.shape geometry args.scalar.grid hr hb hl active
  exact ⟨pointer.1,pointer.2.2⟩

/-- The upstream pointer uses its actual snapshotted stride. Dense saved
labels and saved LSE use `dense_read_pointer` with widths eight and four. -/
theorem backward_upstream_pointer {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {objects : ObjectRegistry rt.toRuntime} {c : Context}
    {upstream output : Tensor} {initial final : World α σ}
    {b : BackwardData M rt c upstream output initial final} (k : BackwardKernelData obs objects b)
    (args : BackwardArguments layout rights k) (shape : upstream.shape=[b.rows]) (row : Nat) (hr : row<b.rows) :
    (Native.address (layout.buffer k.launch.ready.native b.upstream.region).base 4
      (BitVec.ofNat 64 (b.parameters.upstreamBase+row*b.parameters.upstreamStride))).toNat=
      (layout.buffer k.launch.ready.native b.upstream.region).base+4*(b.parameters.upstreamBase+row*b.parameters.upstreamStride) ∧
    ∀ byte, byte<4 → rights.readable k.launch.ready.native b.upstream.device
      ((layout.buffer k.launch.ready.native b.upstream.region).base+4*(b.parameters.upstreamBase+row*b.parameters.upstreamStride)+byte) :=
  vector_pointer layout rights k.launch.ready.native b.upstream 4 b.rows row args.upstream (by decide)
    (k.snapshot.materialized.metadata.shape.trans shape) k.snapshot.materialized.rank hr

/-- The other two backward reads use the actual saved dense objects. The
label role is eight bytes; the global saved LSE role is four bytes. -/
theorem backward_saved_pointers {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {objects : ObjectRegistry rt.toRuntime} {c : Context}
    {upstream output : Tensor} {initial final : World α σ}
    {b : BackwardData M rt c upstream output initial final} (k : BackwardKernelData obs objects b)
    (args : BackwardArguments layout rights k) (labelShape : c.labels.shape=[b.rows]) (lseShape : c.lse.shape=[b.rows])
    (labelDense : c.labels.Dense) (lseDense : c.lse.Dense) (row : Nat) (hr : row<b.rows) :
    (∀ byte, byte<8 → rights.readable k.launch.ready.native c.labels.device
      ((Native.address (layout.buffer k.launch.ready.native c.labels.region).base 8
        (BitVec.ofNat 64 (b.parameters.labelsBase+row))).toNat+byte)) ∧
    (∀ byte, byte<4 → rights.readable k.launch.ready.native c.lse.device
      ((Native.address (layout.buffer k.launch.ready.native c.lse.region).base 4
        (BitVec.ofNat 64 (b.parameters.lseBase+row))).toNat+byte)) := by
  have labels := dense_read_pointer layout rights k.launch.ready.native c.labels 8 b.rows row args.labels (by decide) labelShape labelDense hr
  have lse := dense_read_pointer layout rights k.launch.ready.native c.lse 4 b.rows row args.lse (by decide) lseShape lseDense hr
  constructor
  · intro byte hbyte
    change rights.readable k.launch.ready.native c.labels.device
      ((Native.address (layout.buffer k.launch.ready.native c.labels.region).base 8 (BitVec.ofNat 64 (c.labels.storageOffset+row))).toNat+byte)
    rw [labels.1]
    exact labels.2 byte hbyte
  · intro byte hbyte
    change rights.readable k.launch.ready.native c.lse.device
      ((Native.address (layout.buffer k.launch.ready.native c.lse.region).base 4 (BitVec.ofNat 64 (c.lse.storageOffset+row))).toNat+byte)
    rw [lse.1]
    exact lse.2 byte hbyte

/-- Ordinary successful binder/launch limits. They describe the actual API
outcome, including its signed-i64 conversion; they impose no caller size cap.
Descriptor backing and permissions remain ordinary tensor-object contracts. -/
structure SetupABI {α σ : Type} [Inhabited α] (rt : Runtime α σ) : Prop where
  forward : ∀ p n before ready, rt.setupForwardOps p n before=.ok () ready →
    ScalarABI p (CrossEntropy1.Math.blocks p.cols p.block)
  backward : ∀ p n before ready, rt.setupBackwardOps p n before=.ok () ready →
    ScalarABI p (CrossEntropy1.Math.blocks p.cols p.block)

theorem forward_arguments {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} (call : Entry.Invocation obs M) (setup : SetupABI call.runtime)
    (logits : Descriptor layout rights call.kernel.launch.ready.native call.data.workingLogits 4)
    (labels : Descriptor layout rights call.kernel.launch.ready.native call.data.labels 8)
    (loss : WritableDescriptor layout rights call.kernel.launch.ready.native call.data.losses 4)
    (lse : WritableDescriptor layout rights call.kernel.launch.ready.native call.data.lse 4)
    (zloss : WritableDescriptor layout rights call.kernel.launch.ready.native call.data.zlosses 4) :
    ForwardArguments layout rights call.kernel :=
  ⟨setup.forward _ _ _ _ call.kernel.launch.setupCall,logits,labels,loss,lse,zloss⟩

theorem backward_arguments {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {objects : ObjectRegistry rt.toRuntime} {c : Context}
    {upstream output : Tensor} {initial final : World α σ}
    {b : BackwardData M rt c upstream output initial final} (k : BackwardKernelData obs objects b) (setup : SetupABI rt)
    (logits : Descriptor layout rights k.launch.ready.native c.logits 4)
    (labels : Descriptor layout rights k.launch.ready.native c.labels 8)
    (lse : Descriptor layout rights k.launch.ready.native c.lse 4)
    (upstream : Descriptor layout rights k.launch.ready.native b.upstream 4)
    (gradient : WritableDescriptor layout rights k.launch.ready.native b.work 4) : BackwardArguments layout rights k :=
  ⟨setup.backward _ _ _ _ k.launch.setupCall,logits,labels,lse,upstream,gradient⟩

#axiomsClean backward_logit_pointer
#axiomsClean backward_upstream_pointer
#axiomsClean backward_saved_pointers
#axiomsClean forward_arguments
#axiomsClean backward_arguments
#specNonCircular SetupABI avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Physical
