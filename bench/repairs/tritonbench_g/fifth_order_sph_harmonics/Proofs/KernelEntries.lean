import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.KernelScalar
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.KernelTensor
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Lowering

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.KernelEntries
open VeriTile.Triton FP.Structural Host KernelTensor
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 64000
set_option linter.unusedSimpArgs false

/-- Prepared forward metadata and actual shared Tensor bindings. No numerical
result or execution premise occurs in this ordinary interface. -/
structure Forward where
  request : Request
  admitted : Admitted request
  allocated : TensorObject
  allocation : Allocation request allocated
  coordinates : Binding request.coordinates.metadata
  output : Binding (request.output allocated).metadata

def Forward.params (f : Forward) := parameters f.request f.admitted f.allocated
def Forward.buffers (f : Forward) := forwardBuffers f.request f.allocated

/-- Backward keeps arbitrary supported upstream strides. FreshGradient supplies
only the ordinary prepared dense destination metadata and storage separation. -/
structure Backward (f : Forward) where
  upstreamObject : TensorObject
  gradientObject : TensorObject
  valid : UpstreamContract f.request.coordinates.metadata upstreamObject.metadata
    (f.request.output f.allocated).metadata.columns
  fresh : FreshGradient f.request.coordinates.metadata upstreamObject.metadata gradientObject
  upstream : Binding upstreamObject.metadata
  gradient : Binding gradientObject.metadata

def Backward.params (w : Backward f) := backwardParameters f.params w.upstreamObject.metadata
def Backward.buffers (w : Backward f) :=
  backwardBuffers (save f.request f.allocated) w.upstreamObject w.gradientObject

theorem fit (f : Forward) : f.params.columnOffset+11≤f.params.outputWidth :=
  parameters_fit f.request f.admitted f.allocated f.allocation

theorem upstream_rows (w : Backward f) : w.upstreamObject.metadata.rows=f.params.rows := by
  simp [TensorView.rows,w.valid.2.1,Forward.params,parameters]
theorem upstream_columns (w : Backward f) : w.upstreamObject.metadata.columns=f.params.outputWidth := by
  simp [TensorView.columns,w.valid.2.1,Forward.params,parameters]
theorem gradient_coordinates (w : Backward f) : Coordinates w.gradientObject.metadata :=
  ⟨w.fresh.kind,w.fresh.columns,w.fresh.strides,w.fresh.bounds⟩

/-- Read a tagged source state through the actual destination Tensor's physical
index relative to its queried data pointer, including storage offsets. -/
def tensorCell (state : State α) (region : RegionName) (b : Binding view) (r c : Nat) : Cell α :=
  state.mem region (b.tensor.physicalIndex (r*view.columns+c)-b.tensor.storageOffset)

theorem output_index (f : Forward) (r : Nat) (hr : r<f.params.rows) (c : Fin 11) :
    f.output.tensor.physicalIndex (r*(f.request.output f.allocated).metadata.columns+
      (f.params.columnOffset+c.val))-f.output.tensor.storageOffset=
      r*f.params.outputWidth+f.params.columnOffset+c.val := by
  have dest:=destination f.request f.admitted f.allocated f.allocation
  have rows:=dest.2.1
  have hc:=c.isLt
  have hf:=fit f
  rw [f.output.strides r (by change r<f.request.coordinates.metadata.rows at hr;omega)
    (f.params.columnOffset+c.val) (by change f.params.columnOffset+11≤(f.request.output f.allocated).metadata.columns at hf;omega)]
  simp only [TensorView.rowStride,TensorView.colStride,dest.2.2.2.2.1]
  simp [Forward.params,parameters,Nat.add_assoc]

theorem gradient_index (w : Backward f) (r : Nat) (hr : r<f.params.rows) (c : Fin 3) :
    w.gradient.tensor.physicalIndex (r*w.gradientObject.metadata.columns+c.val)-
      w.gradient.tensor.storageOffset=r*3+c.val := by
  have rows:=w.fresh.rows
  rw [w.gradient.strides r (by change r<f.request.coordinates.metadata.rows at hr;omega)
    c.val (by rw [w.fresh.columns];exact c.isLt)]
  simp [TensorView.rowStride,TensorView.colStride,w.fresh.strides,Nat.add_assoc]

theorem output_address (f : Forward) (r : Nat) (hr : r<f.params.rows) (c : Fin 11) :
    f.buffers.destination+4*(r*f.params.outputWidth+f.params.columnOffset+c.val)=
      f.output.tensor.storage.address (f.output.tensor.physicalIndex
        (r*(f.request.output f.allocated).metadata.columns+(f.params.columnOffset+c.val))) := by
  have dest:=destination f.request f.admitted f.allocated f.allocation
  have hf:=fit f
  have hc:=c.isLt
  have address:=KernelTensor.address f.output dest.1 r (f.params.columnOffset+c.val)
    (by have rows:=dest.2.1;change r<f.request.coordinates.metadata.rows at hr;omega)
    (by change f.params.columnOffset+11≤(f.request.output f.allocated).metadata.columns at hf;omega)
  rw [Access.destination_address _ _ _ dest] at address
  simpa [Forward.buffers,forwardBuffers,Forward.params,parameters,Nat.add_assoc] using address

theorem gradient_address (w : Backward f) (r : Nat) (hr : r<f.params.rows) (c : Fin 3) :
    w.buffers.destination+4*(r*3+c.val)=w.gradient.tensor.storage.address
      (w.gradient.tensor.physicalIndex (r*w.gradientObject.metadata.columns+c.val)) := by
  have address:=KernelTensor.address w.gradient w.fresh.kind r c.val
    (by have rows:=w.fresh.rows;change r<f.request.coordinates.metadata.rows at hr;omega)
    (by rw [w.fresh.columns];exact c.isLt)
  rw [Access.coordinate_address _ (gradient_coordinates w)] at address
  exact address

/-- The finite lowered source pointer is the actual shared output Tensor
address; all PID/lane extension and modular arithmetic are discharged below. -/
theorem lowered_output_pointer (f : Forward) (pid lane : Nat) (c : Fin 11)
    (hp : pid<Grid.blocks f.params) (hl : lane<f.params.blockSize)
    (active : pid*f.params.blockSize+lane<f.params.rows) :
    Native.pointer f.buffers.destination f.params.blockSize f.params.outputWidth 1
      f.params.columnOffset pid lane c.val=
      f.output.tensor.storage.address (f.output.tensor.physicalIndex
        ((pid*f.params.blockSize+lane)*(f.request.output f.allocated).metadata.columns+
          (f.params.columnOffset+c.val))) :=
  (Lowering.forward_pointer f.request f.admitted f.allocated f.allocation pid lane c hp hl active).trans
    (output_address f _ active c)

theorem lowered_gradient_pointer (w : Backward f) (pid lane : Nat) (c : Fin 3)
    (hp : pid<Grid.blocks f.params) (hl : lane<f.params.blockSize)
    (active : pid*f.params.blockSize+lane<f.params.rows) :
    Native.pointer w.buffers.destination f.params.blockSize 3 1 0 pid lane c.val=
      w.gradient.tensor.storage.address (w.gradient.tensor.physicalIndex
        ((pid*f.params.blockSize+lane)*w.gradientObject.metadata.columns+c.val)) :=
  (Lowering.gradient_pointer f.request f.admitted f.allocated f.allocation w.upstreamObject.metadata
    w.gradientObject w.fresh pid lane c hp hl active).trans (gradient_address w _ active c)

theorem lowered_coordinate_pointer (f : Forward) (pid lane : Nat) (c : Fin 3)
    (hp : pid<Grid.blocks f.params) (hl : lane<f.params.blockSize)
    (active : pid*f.params.blockSize+lane<f.params.rows) :
    Native.pointer f.buffers.coordinates f.params.blockSize 3 1 0 pid lane c.val=
      f.coordinates.tensor.storage.address (f.coordinates.tensor.physicalIndex
        ((pid*f.params.blockSize+lane)*f.request.coordinates.metadata.columns+c.val)) := by
  refine (Lowering.coordinate_pointer f.request f.admitted f.allocated f.allocation pid lane c hp hl active).trans ?_
  rw [←Access.coordinate_address _ f.admitted.1]
  exact KernelTensor.address f.coordinates f.admitted.1.1 _ c.val active
    (by rw [f.admitted.1.2.1];exact c.isLt)

theorem lowered_upstream_pointer (w : Backward f) (pid lane : Nat) (c : Fin 11)
    (hp : pid<Grid.blocks f.params) (hl : lane<f.params.blockSize)
    (active : pid*f.params.blockSize+lane<f.params.rows) :
    Native.pointer w.buffers.upstream f.params.blockSize w.params.upstreamRowStride w.params.upstreamColStride
      f.params.columnOffset pid lane c.val=
      w.upstream.tensor.storage.address (w.upstream.tensor.physicalIndex
        ((pid*f.params.blockSize+lane)*w.upstreamObject.metadata.columns+(f.params.columnOffset+c.val))) := by
  refine (Lowering.upstream_pointer f.request f.admitted f.allocated f.allocation w.upstreamObject.metadata
    w.valid pid lane c hp hl active).trans ?_
  exact KernelTensor.address w.upstream w.valid.1 _ _ (by rw [upstream_rows w];exact active)
    (by rw [upstream_columns w];change f.params.columnOffset+c.val<f.params.outputWidth
        have hf:=fit f;have hc:=c.isLt;omega)

def realForward (f : Forward) (memory : Nat → ℝ) (r : Nat) : Fin 11 → ℝ := fun c=>
  Polynomial.value c (realPoint memory f.coordinates r 0) (realPoint memory f.coordinates r 1)
    (realPoint memory f.coordinates r 2)
def realBackward (w : Backward f) (memory : Nat → ℝ) (r : Nat) : Fin 3 → ℝ :=
  Polynomial.vjp (realPoint memory f.coordinates r 0) (realPoint memory f.coordinates r 1)
    (realPoint memory f.coordinates r 2)
    (fun c=>realPoint memory w.upstream r (f.params.columnOffset+c.val))
def nativeForward (f : Forward) (heap : Storage.Heap) (r : Nat) : Fin 11 → FP.Value .fp32 :=
  Software.forward KernelScalar.selectedProfile (nativePoint heap f.coordinates r 0)
    (nativePoint heap f.coordinates r 1) (nativePoint heap f.coordinates r 2)
/-- The initial +0 addition is retained, as in source, even when its effect on
signed zero differs from dropping the accumulation. -/
def nativeBackward (w : Backward f) (heap : Storage.Heap) (r : Nat) : Fin 3 → FP.Value .fp32 :=
  Software.backward KernelScalar.selectedProfile (nativePoint heap f.coordinates r 0)
    (nativePoint heap f.coordinates r 1) (nativePoint heap f.coordinates r 2)
    (fun c=>nativePoint heap w.upstream r (f.params.columnOffset+c.val))
    (fun _=>FP.Value.zero .fp32)

def RealPrepared (w : Backward f) (memory : Nat → ℝ) : Prop :=
  ∀ r,r<f.params.rows → ∀ c : Fin 3,realPoint memory w.gradient r c.val=0
def NativePrepared (w : Backward f) (heap : Storage.Heap) : Prop :=
  ∀ r,r<f.params.rows → ∀ c : Fin 3,nativePoint heap w.gradient r c.val=FP.Value.zero .fp32

theorem real_forward_reference (f : Forward) (memory : Nat → ℝ) (r : Nat) (hr : r<f.params.rows) :
    KernelScalar.realForward f.buffers memory r=realForward f memory r := by
  funext c
  simp only [KernelScalar.realForward,realForward,Forward.buffers,forwardBuffers]
  have h0:=KernelTensor.coordinate_real memory f.coordinates f.admitted.1 r hr 0
  have h1:=KernelTensor.coordinate_real memory f.coordinates f.admitted.1 r hr 1
  have h2:=KernelTensor.coordinate_real memory f.coordinates f.admitted.1 r hr 2
  simp only [Fin.val_zero,Fin.val_one,Fin.val_two,Nat.add_zero] at h0 h1 h2 ⊢
  rw [h0,h1,h2]

theorem real_upstream (w : Backward f) (memory : Nat → ℝ) (r : Nat) (hr : r<f.params.rows) (c : Fin 11) :
    memory (w.buffers.upstream+4*(r*w.params.upstreamRowStride+
      (w.params.columnOffset+c.val)*w.params.upstreamColStride))=
      realPoint memory w.upstream r (f.params.columnOffset+c.val) := by
  exact KernelTensor.real_read memory w.upstream w.valid.1 r (f.params.columnOffset+c.val)
    (by rw [upstream_rows w];exact hr) (by rw [upstream_columns w];have hf:=fit f;have hc:=c.isLt;omega)

theorem real_backward_reference (w : Backward f) (memory : Nat → ℝ) (r : Nat) (hr : r<f.params.rows) :
    KernelScalar.realBackward w.params w.buffers memory r=realBackward w memory r := by
  unfold KernelScalar.realBackward realBackward
  have coords (c : Fin 3) := KernelTensor.coordinate_real memory f.coordinates f.admitted.1 r hr c
  simp only [Backward.buffers,backwardBuffers,save] at *
  have h0:=coords 0
  have h1:=coords 1
  have h2:=coords 2
  simp only [Fin.val_zero,Fin.val_one,Fin.val_two,Nat.add_zero] at h0 h1 h2 ⊢
  rw [h0,h1,h2]
  congr 1
  funext c
  exact real_upstream w memory r hr c

theorem real_zero (w : Backward f) (memory : Nat → ℝ) (zero : RealPrepared w memory)
    (r : Nat) (hr : r<f.params.rows) (c : Fin 3) : memory (w.buffers.destination+4*(r*3+c.val))=0 := by
  exact (KernelTensor.coordinate_real memory w.gradient (gradient_coordinates w) r
    (by have rows:=w.fresh.rows;change r<f.request.coordinates.metadata.rows at hr;omega) c).trans (zero r hr c)

theorem native_forward_reference (f : Forward) (heap : Storage.Heap) (r : Nat) (hr : r<f.params.rows) :
    KernelScalar.forwardReference KernelScalar.selectedProfile f.buffers heap r=nativeForward f heap r := by
  unfold KernelScalar.forwardReference nativeForward Equivalence.forwardInputs
  simp only [Forward.buffers,forwardBuffers]
  have h0:=KernelTensor.coordinate_native heap f.coordinates f.admitted.1 r hr 0
  have h1:=KernelTensor.coordinate_native heap f.coordinates f.admitted.1 r hr 1
  have h2:=KernelTensor.coordinate_native heap f.coordinates f.admitted.1 r hr 2
  simp only [Fin.val_zero,Fin.val_one,Fin.val_two,Nat.add_zero] at h0 h1 h2 ⊢
  rw [h0,h1,h2]

theorem native_upstream (w : Backward f) (heap : Storage.Heap) (r : Nat) (hr : r<f.params.rows) (c : Fin 11) :
    Packing.value (Equivalence.backwardInputs w.params w.buffers heap r (3+c.val))=
      nativePoint heap w.upstream r (f.params.columnOffset+c.val) := by
  rw [Equivalence.input_upstream]
  exact KernelTensor.native_read heap w.upstream w.valid.1 r (f.params.columnOffset+c.val)
    (by rw [upstream_rows w];exact hr) (by rw [upstream_columns w];have hf:=fit f;have hc:=c.isLt;omega)

theorem native_zero (w : Backward f) (heap : Storage.Heap) (zero : NativePrepared w heap)
    (r : Nat) (hr : r<f.params.rows) (c : Fin 3) :
    Packing.value (Equivalence.backwardInputs w.params w.buffers heap r (14+c.val))=FP.Value.zero .fp32 := by
  rw [Equivalence.input_gradient]
  exact (KernelTensor.coordinate_native heap w.gradient (gradient_coordinates w) r
    (by have rows:=w.fresh.rows;change r<f.request.coordinates.metadata.rows at hr;omega) c).trans (zero r hr c)

theorem native_backward_reference (w : Backward f) (heap : Storage.Heap) (zero : NativePrepared w heap)
    (r : Nat) (hr : r<f.params.rows) :
    KernelScalar.backwardReference KernelScalar.selectedProfile w.params w.buffers heap r=nativeBackward w heap r := by
  unfold KernelScalar.backwardReference nativeBackward
  have hg : (fun c : Fin 11=>Packing.value (Equivalence.backwardInputs w.params w.buffers heap r (3+c.val)))=
      (fun c=>nativePoint heap w.upstream r (f.params.columnOffset+c.val)) := funext (native_upstream w heap r hr)
  have hz : (fun c : Fin 3=>Packing.value (Equivalence.backwardInputs w.params w.buffers heap r (14+c.val)))=
      (fun _=>FP.Value.zero .fp32) := funext (native_zero w heap zero r hr)
  rw [hg,hz]
  simp only [Equivalence.backwardInputs,show (0:Nat)<3 by omega,show (1:Nat)<3 by omega,
    show (2:Nat)<3 by omega,ite_true,Equivalence.forwardInputs,Backward.buffers,backwardBuffers,save]
  have h0:=KernelTensor.coordinate_native heap f.coordinates f.admitted.1 r hr 0
  have h1:=KernelTensor.coordinate_native heap f.coordinates f.admitted.1 r hr 1
  have h2:=KernelTensor.coordinate_native heap f.coordinates f.admitted.1 r hr 2
  simp only [Fin.val_zero,Fin.val_one,Fin.val_two,Nat.add_zero] at h0 h1 h2 ⊢
  rw [h0,h1,h2]

#axiomsClean output_address
#axiomsClean gradient_address
#axiomsClean lowered_output_pointer
#axiomsClean lowered_gradient_pointer
#axiomsClean lowered_coordinate_pointer
#axiomsClean lowered_upstream_pointer
#axiomsClean real_forward_reference
#axiomsClean real_backward_reference
#axiomsClean native_forward_reference
#axiomsClean native_backward_reference
#specNonCircular realForward avoiding [forwardKernel,backwardKernel,Grid.launch]
#specNonCircular realBackward avoiding [forwardKernel,backwardKernel,Grid.launch]
#specNonCircular nativeForward avoiding [forwardKernel,backwardKernel,Grid.launch]
#specNonCircular nativeBackward avoiding [forwardKernel,backwardKernel,Grid.launch]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.KernelEntries
