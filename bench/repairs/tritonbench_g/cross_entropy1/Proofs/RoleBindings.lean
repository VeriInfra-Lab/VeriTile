import bench.repairs.tritonbench_g.cross_entropy1.Proofs.LaunchBinding
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Entries
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.MaskBytes

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

theorem dense_read_pointer {σ : Type} (layout : Layout σ) (rights : Permissions σ) (native : σ)
    (t : Tensor) (width rows row : Nat) (descriptor : Descriptor layout rights native t width)
    (positive : 0<width) (shape : t.shape=[rows]) (dense : t.Dense) (inside : row<rows) :
    (Native.address (layout.buffer native t.region).base width (BitVec.ofNat 64 (t.storageOffset+row))).toNat=
      (layout.buffer native t.region).base+width*(t.storageOffset+row) ∧
    ∀ byte, byte<width → rights.readable native t.device
      ((layout.buffer native t.region).base+width*(t.storageOffset+row)+byte) := by
  have bound : row<t.numel := by simpa [Tensor.numel,shape] using inside
  have base := dense_base t dense (by omega)
  have coordinate : t.index row=t.storageOffset+row := by rw [dense.2 row bound,base,Nat.zero_add]
  have address := descriptor_address layout rights native t width descriptor positive row bound
  exact ⟨by simpa only [coordinate] using address.1,
    fun byte hbyte => by simpa only [coordinate] using descriptor.readable row byte bound hbyte⟩

/-- All three scalar/vector forward roles use the actual launch tensor
objects. The logit role is `forward_logit_pointer` in LaunchBinding. -/
theorem forward_scalar_roles {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} (call : Entry.Invocation obs M)
    (args : ForwardArguments layout rights call.kernel) (row block : Nat)
    (hr : row<call.data.rows) (hb : block<call.data.count) :
    (Native.address (layout.buffer call.kernel.launch.ready.native call.data.labels.region).base 8
      (BitVec.ofNat 64 (call.data.parameters.labelsBase+row))).toNat=
      (layout.buffer call.kernel.launch.ready.native call.data.labels.region).base+8*(call.data.parameters.labelsBase+row) ∧
    (∀ byte, byte<8 → rights.readable call.kernel.launch.ready.native call.data.labels.device
      ((layout.buffer call.kernel.launch.ready.native call.data.labels.region).base+8*(call.data.parameters.labelsBase+row)+byte)) ∧
    (∀ t, t=call.data.losses ∨ t=call.data.lse →
      (Native.address (layout.buffer call.kernel.launch.ready.native t.region).base 4
        (Native.scalarOutput t.storageOffset call.data.rows row block)).toNat=
        (layout.buffer call.kernel.launch.ready.native t.region).base+4*(t.storageOffset+block*call.data.rows+row) ∧
      ∀ byte, byte<4 → rights.writable call.kernel.launch.ready.native t.device
        ((layout.buffer call.kernel.launch.ready.native t.region).base+4*(t.storageOffset+block*call.data.rows+row)+byte)) := by
  have labels := dense_read_pointer layout rights call.kernel.launch.ready.native call.data.labels 8 call.data.rows row
    args.labels (by decide) (call.kernel.labelsSnapshot.materialized.metadata.shape.trans call.data.labelsShape)
    call.kernel.labelsSnapshot.dense hr
  refine ⟨labels.1,labels.2,?_⟩
  intro t choice
  rcases choice with rfl|rfl
  · exact scalar_store_pointer layout rights call.kernel.launch.ready.native call.data.losses call.data.count call.data.rows
      call.data.parameters.block call.data.cols block row args.loss args.scalar.grid call.kernel.lossDense call.kernel.lossAllocation.shapeEq hb hr
  · exact scalar_store_pointer layout rights call.kernel.launch.ready.native call.data.lse call.data.count call.data.rows
      call.data.parameters.block call.data.cols block row args.lse args.scalar.grid call.kernel.lseDense call.kernel.lseAllocation.shapeEq hb hr

theorem backward_logit_pointer {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {objects : ObjectRegistry rt} {c : Context}
    {upstream output : Tensor} {initial final : World α σ}
    {b : BackwardData M rt c upstream output initial final} (k : BackwardKernelData obs objects b)
    (args : BackwardArguments layout rights k) (geometry : RowLayout c.logits) (row block lane : Nat)
    (hr : row<b.rows) (hb : block<b.count) (hl : lane<b.parameters.block)
    (active : column b.parameters block lane<b.cols) :
    (Native.address (layout.buffer k.launch.ready.native c.logits.region).base 4
      (Native.element b.parameters.logitsBase row b.parameters.logitsStride b.parameters.block block lane)).toNat=
      (layout.buffer k.launch.ready.native c.logits.region).base+4*logitAddress b.parameters row (column b.parameters block lane) ∧
    ∀ byte, byte<4 → rights.readable k.launch.ready.native c.logits.device
      ((layout.buffer k.launch.ready.native c.logits.region).base+4*logitAddress b.parameters row (column b.parameters block lane)+byte) := by
  have pointer := strided_address layout rights k.launch.ready.native c.logits args.logits
    b.rows b.cols b.parameters.block b.count row block lane b.shape geometry args.scalar.grid hr hb hl active
  exact ⟨pointer.1,pointer.2.2⟩

/-- The upstream pointer uses its actual snapshotted stride. Dense saved
labels and saved LSE use `dense_read_pointer` with widths eight and four. -/
theorem backward_upstream_pointer {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {objects : ObjectRegistry rt} {c : Context}
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

/-- The source target-gather branch itself proves a backed logical column;
neither block padding nor a fabricated tail allocation is needed. -/
theorem target_coordinate (target start : Int) (cols width block : Nat)
    (branch : (block*width : Nat)≤target-start ∧ target-start<(min cols ((block+1)*width) : Nat)) :
    0≤target-start ∧ (target-start).toNat<cols := by
  have upper : min cols ((block+1)*width)≤cols := Nat.min_le_left _ _
  constructor
  · omega
  · omega

theorem forward_target_pointer {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    {obs : Observations α σ} {M : FP32 α} (call : Entry.Invocation obs M)
    (args : ForwardArguments layout rights call.kernel) (row block : Nat) (hr : row<call.data.rows)
    (target : Int) (branch : (block*call.data.parameters.block : Nat)≤target-call.data.parameters.classStart ∧
      target-call.data.parameters.classStart<(min call.data.cols ((block+1)*call.data.parameters.block) : Nat)) :
    let col := (target-call.data.parameters.classStart).toNat
    (Native.address (layout.buffer call.kernel.launch.ready.native call.data.workingLogits.region).base 4
      (BitVec.ofNat 64 (logitAddress call.data.parameters row col))).toNat=
      (layout.buffer call.kernel.launch.ready.native call.data.workingLogits.region).base+4*logitAddress call.data.parameters row col ∧
    ∀ byte, byte<4 → rights.readable call.kernel.launch.ready.native call.data.workingLogits.device
      ((layout.buffer call.kernel.launch.ready.native call.data.workingLogits.region).base+4*logitAddress call.data.parameters row col+byte) := by
  have col := (target_coordinate target call.data.parameters.classStart call.data.cols call.data.parameters.block block branch).2
  have shape := call.kernel.working.metadata.shape.trans call.data.logitsShape
  have bound := row_coordinate_bound call.data.workingLogits call.data.rows call.data.cols row _ shape hr col
  have coordinate := call.kernel.layout call.data.rows call.data.cols shape row _ hr col
  have address := descriptor_address layout rights call.kernel.launch.ready.native call.data.workingLogits 4 args.logits (by decide) _ bound
  exact ⟨by simpa only [coordinate] using address.1,
    fun byte hbyte => by simpa only [coordinate] using args.logits.readable _ byte bound hbyte⟩

/-- Local annotated signed-i64 conversion, including the failure boundary.
It is a scalar ABI check and does not posit execution success. -/
def scalar64 (x : Int) : Except String (BitVec 64) :=
  if -(2^63 : Int)≤x ∧ x<2^63 then .ok (BitVec.ofInt 64 x) else .error "signed int64 conversion"

theorem scalar64_success (x : Int) (inside : -(2^63 : Int)≤x ∧ x<2^63) :
    scalar64 x=.ok (BitVec.ofInt 64 x) ∧ (BitVec.ofInt 64 x).toInt=x := by
  refine ⟨by simp only [scalar64,if_pos inside],?_⟩
  apply BitVec.toInt_ofInt_eq_self <;> norm_num at inside ⊢ <;> omega

theorem scalar64_failure (x : Int) (outside : ¬(-(2^63 : Int)≤x ∧ x<2^63)) :
    scalar64 x=.error "signed int64 conversion" := by simp only [scalar64,if_neg outside]

#axiomsClean forward_scalar_roles
#axiomsClean backward_logit_pointer
#axiomsClean backward_upstream_pointer
#axiomsClean target_coordinate
#axiomsClean forward_target_pointer
#axiomsClean scalar64_success
#axiomsClean scalar64_failure
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
