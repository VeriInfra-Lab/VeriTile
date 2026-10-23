import bench.repairs.tritonbench_g.lightning_attention.Proofs.HostIngress
import bench.repairs.tritonbench_g.lightning_attention.Proofs.AddressBounds
import bench.repairs.tritonbench_g.lightning_attention.Proofs.Ownership
namespace VeriTile.Bench.Repairs.LightningAttention.Host
open VeriTile Triton FP.Structural ChunkLinearAttn
noncomputable section
set_option maxHeartbeats 6000000

def buffer (r : RegionName) (t : Tensor) : Physical.Buffer := ⟨r,t.base,t.shape.prod⟩
def forwardBuffers (a : Allocations) : List Physical.Buffer := [buffer "output" a.output]
def backwardBuffers (a : Allocations) : List Physical.Buffer :=
  [buffer "queryGradient" a.queryGradient,buffer "keyGradient" a.keyGradient,
   buffer "valueGradient" a.valueGradient]
def outputBuffers (a : Allocations) : List Physical.Buffer := forwardBuffers a++backwardBuffers a
def protectedTensors (c : Invocation) (a : Allocations) : List Tensor := c.inputs++a.inputs
def inputBuffers (c : Invocation) (a : Allocations) : List Physical.Buffer :=
  (protectedTensors c a).map (fun t => ⟨"protectedInput",t.base,span t⟩)
def deviceBuffers (a : Allocations) : List Physical.Buffer :=
  [buffer "query" a.queryDense,buffer "key" a.keyDense,buffer "value" a.valueDense,
   buffer "incoming" a.incomingDense]++outputBuffers a
def forwardProtected (c : Invocation) (a : Allocations) : List Tensor :=
  [c.query.tensor,c.key.tensor,c.value.tensor,a.queryDense,a.keyDense,a.valueDense]
def backwardProtected (c : Invocation) (a : Allocations) : List Tensor :=
  protectedTensors c a++[a.output]
def forwardDeviceBuffers (a : Allocations) : List Physical.Buffer :=
  [buffer "query" a.queryDense,buffer "key" a.keyDense,buffer "value" a.valueDense]++forwardBuffers a
def backwardDeviceBuffers (a : Allocations) : List Physical.Buffer :=
  [buffer "query" a.queryDense,buffer "key" a.keyDense,buffer "value" a.valueDense,
   buffer "incoming" a.incomingDense]++backwardBuffers a

/-- Metadata for publication of any chosen list of output buffers. In the
separate public calls, forward and backward use their own temporal input sets. -/
structure PublicationAllocation (outputs : List Physical.Buffer) (inputs : List Tensor) : Prop where
  separate : ∀ x∈outputs, ∀ y∈outputs, x≠y → Physical.Separate x y
  inputFresh : ∀ t∈inputs, ∀ y∈outputs, Physical.Separate ⟨"protectedInput",t.base,span t⟩ y

/-- Forward allocator facts concern only the inputs present at forward time. -/
structure ForwardPhysicalAllocation (c : Invocation) (a : Allocations) : Prop where
  publication : PublicationAllocation (forwardBuffers a) (forwardProtected c a)
  pointerRoom : PointerABI c.dims (forwardDeviceBuffers a)
/-- Backward allocates gradients after the incoming cotangent is supplied.
The cotangent may alias output or any saved input; only gradients are fresh. -/
structure BackwardPhysicalAllocation (c : Invocation) (a : Allocations) : Prop where
  publication : PublicationAllocation (backwardBuffers a) (backwardProtected c a)
  pointerRoom : PointerABI c.dims (backwardDeviceBuffers a)

/-- Combined-call physical layout. Stage-wise clients may use the weaker
PublicationAllocation for each stage, allowing a later cotangent to alias O. -/
structure PhysicalAllocation (c : Invocation) (a : Allocations) : Prop where
  publication : PublicationAllocation (outputBuffers a) (protectedTensors c a)
  pointerRoom : PointerABI c.dims (deviceBuffers a)

def publish {α : Type} (a : Allocations) (values : RegionName → Nat → α)
    (before : Nat → α) : Nat → α := Physical.publish (outputBuffers a) values before
def publishForward {α : Type} (a : Allocations) (values : RegionName → Nat → α)
    (before : Nat → α) : Nat → α := Physical.publish (forwardBuffers a) values before
def publishBackward {α : Type} (a : Allocations) (values : RegionName → Nat → α)
    (before : Nat → α) : Nat → α := Physical.publish (backwardBuffers a) values before

theorem publication_read {α : Type} (outputs : List Physical.Buffer) (inputs : List Tensor)
    (pa : PublicationAllocation outputs inputs) (values : RegionName → Nat → α) (before : Nat → α)
    (b : Physical.Buffer) (hb : b∈outputs) (o : Nat) (ho : o<b.words) :
    Physical.publish outputs values before (b.base+o)=values b.region o :=
  Physical.publish_read outputs values before b hb o ho
    (fun y hy hn => pa.separate b hb y hy (Ne.symm hn))
theorem publication_inputs {α : Type} (outputs : List Physical.Buffer) (inputs : List Tensor)
    (pa : PublicationAllocation outputs inputs) (values : RegionName → Nat → α) (before : Nat → α)
    (t : Tensor) (ht : t∈inputs) (o : Nat) (ho : o<span t) :
    Physical.publish outputs values before (t.base+o)=before (t.base+o) :=
  Physical.publish_input outputs values before ⟨"protectedInput",t.base,span t⟩ (pa.inputFresh t ht) o ho

theorem forward_published_read {α : Type} (c : Invocation) (a : Allocations)
    (pa : ForwardPhysicalAllocation c a) (values : RegionName → Nat → α) (before : Nat → α)
    (o : Nat) (ho : o<a.output.shape.prod) :
    publishForward a values before (a.output.base+o)=values "output" o :=
  publication_read _ _ pa.publication values before (buffer "output" a.output) (by simp [forwardBuffers]) o ho
theorem forward_published_inputs {α : Type} (c : Invocation) (a : Allocations)
    (pa : ForwardPhysicalAllocation c a) (values : RegionName → Nat → α) (before : Nat → α)
    (t : Tensor) (ht : t∈forwardProtected c a) (o : Nat) (ho : o<span t) :
    publishForward a values before (t.base+o)=before (t.base+o) :=
  publication_inputs _ _ pa.publication values before t ht o ho
theorem forward_published_frame {α : Type} (a : Allocations) (values : RegionName → Nat → α)
    (before : Nat → α) (address : Nat) (h : ¬Physical.Contains (buffer "output" a.output) address) :
    publishForward a values before address=before address :=
  Physical.publish_frame _ values before address (by simpa [forwardBuffers] using h)
theorem backward_published_read {α : Type} (c : Invocation) (a : Allocations)
    (pa : BackwardPhysicalAllocation c a) (values : RegionName → Nat → α) (before : Nat → α)
    (r : RegionName) (t : Tensor) (ht : buffer r t∈backwardBuffers a)
    (o : Nat) (ho : o<t.shape.prod) :
    publishBackward a values before (t.base+o)=values r o :=
  publication_read _ _ pa.publication values before (buffer r t) ht o ho
theorem backward_published_inputs {α : Type} (c : Invocation) (a : Allocations)
    (pa : BackwardPhysicalAllocation c a) (values : RegionName → Nat → α) (before : Nat → α)
    (t : Tensor) (ht : t∈backwardProtected c a) (o : Nat) (ho : o<span t) :
    publishBackward a values before (t.base+o)=before (t.base+o) :=
  publication_inputs _ _ pa.publication values before t ht o ho
theorem backward_published_frame {α : Type} (a : Allocations) (values : RegionName → Nat → α)
    (before : Nat → α) (address : Nat)
    (h : ∀ b∈backwardBuffers a, ¬Physical.Contains b address) :
    publishBackward a values before address=before address :=
  Physical.publish_frame _ values before address h

theorem two_stage_frame {α : Type} (a : Allocations) (fwd bwd : RegionName → Nat → α)
    (before : Nat → α) (address : Nat) (h : ∀ b∈outputBuffers a, ¬Physical.Contains b address) :
    publishBackward a bwd (publishForward a fwd before) address=before address := by
  rw [backward_published_frame a bwd _ address (fun b hb => h b (List.mem_append_right _ hb))]
  exact forward_published_frame a fwd before address
    (h _ (List.mem_append_left _ (by simp [forwardBuffers])))

theorem published_read {α : Type} (c : Invocation) (a : Allocations) (pa : PhysicalAllocation c a)
    (values : RegionName → Nat → α) (before : Nat → α) (r : RegionName) (t : Tensor)
    (ht : buffer r t∈outputBuffers a) (o : Nat) (ho : o<t.shape.prod) :
    publish a values before (t.base+o)=values r o :=
  publication_read _ _ pa.publication values before (buffer r t) ht o ho
theorem published_inputs {α : Type} (c : Invocation) (a : Allocations) (pa : PhysicalAllocation c a)
    (values : RegionName → Nat → α) (before : Nat → α) (t : Tensor)
    (ht : t∈protectedTensors c a) (o : Nat) (ho : o<span t) :
    publish a values before (t.base+o)=before (t.base+o) :=
  publication_inputs _ _ pa.publication values before t ht o ho
theorem published_frame {α : Type} (a : Allocations) (values : RegionName → Nat → α) (before : Nat → α)
    (address : Nat) (h : ∀ b∈outputBuffers a, ¬Physical.Contains b address) :
    publish a values before address=before address :=
  Physical.publish_frame _ values before address h

theorem output_mem (a : Allocations) : buffer "output" a.output∈outputBuffers a := by
  simp [outputBuffers,forwardBuffers]
theorem query_mem (a : Allocations) : buffer "queryGradient" a.queryGradient∈outputBuffers a := by
  simp [outputBuffers,backwardBuffers]
theorem key_mem (a : Allocations) : buffer "keyGradient" a.keyGradient∈outputBuffers a := by
  simp [outputBuffers,backwardBuffers]
theorem value_mem (a : Allocations) : buffer "valueGradient" a.valueGradient∈outputBuffers a := by
  simp [outputBuffers,backwardBuffers]

theorem output_regions (a : Allocations) :
    (outputBuffers a).map Physical.Buffer.region=
      (["output","queryGradient","keyGradient","valueGradient"] : List RegionName) := rfl
theorem output_sizes (c : Invocation) (a : Allocations) (ac : AllocationContract c a)
    (b : Physical.Buffer) (hb : b∈outputBuffers a) : b.words=regionCapacity c.dims b.region := by
  simp only [outputBuffers,forwardBuffers,backwardBuffers,List.mem_append,List.mem_cons,
    List.not_mem_nil,or_false] at hb
  rcases hb with rfl|(rfl|rfl|rfl)
  all_goals simp [buffer,regionCapacity,ac.outputShape,ac.queryGradientShape,
    ac.keyGradientShape,ac.valueGradientShape,queryShape,valueShape,Nat.mul_assoc]

theorem dense_span_list (shape : List Nat) (positive : ∀ n∈shape, 0<n) :
    ((shape.zip (denseStrides shape)).map (fun z => (z.1-1)*z.2)).sum+1=shape.prod := by
  induction shape with
  | nil => rfl
  | cons n tail ih =>
    have hn : 1≤n := positive n (by simp)
    have ht := ih (fun x hx => positive x (by simp [hx]))
    have hm : (n-1)*tail.prod+tail.prod=n*tail.prod := by
      rw [← Nat.add_one_mul,Nat.sub_add_cancel hn]
    simp only [denseStrides,ChunkDeltaFwd.Host.denseStrides,List.zip_cons_cons,
      List.map_cons,List.sum_cons,List.prod_cons]
    rw [Nat.add_assoc,ht]
    exact hm

theorem layout_span_list (shape strides : List Nat) (positive : ∀ n∈shape, 0<n)
    (layout : DenseLayout shape strides) :
    ((shape.zip strides).map (fun z => (z.1-1)*z.2)).sum+1=shape.prod := by
  induction shape generalizing strides with
  | nil => simp
  | cons n tail ih =>
    cases strides with
    | nil => simp [DenseLayout] at layout
    | cons stride strides =>
      have ht := ih strides (fun x hx => positive x (by simp [hx])) layout.2
      have hn : 1≤n := positive n (by simp)
      simp only [List.zip_cons_cons,List.map_cons,List.sum_cons,List.prod_cons]
      rw [Nat.add_assoc,ht]
      rcases layout.1 with he|he
      · simp [he]
      · rw [he,← Nat.add_one_mul,Nat.sub_add_cancel hn]

theorem allocated_span (c : Invocation) (cc : InvocationContract c) (a : Allocations)
    (ac : AllocationContract c a) (t : Tensor) (ht : t∈a.inputs++a.outputs) : span t=t.shape.prod := by
  have hd := (ac.descriptors t ht).2.2.2.2.1
  have hp : ∀ n∈t.shape, 0<n := by
    have hb := cc.dimensions.b_pos
    have hh := cc.dimensions.h_pos
    have hn := cc.dimensions.n_pos
    have hD : 0<c.dims.D := by have := cc.dimensions.d_min; omega
    have hE : 0<c.dims.E := by have := cc.dimensions.e_min; omega
    simp only [Allocations.inputs,Allocations.outputs,List.mem_append,List.mem_cons,
      List.not_mem_nil,or_false] at ht
    rcases ht with (rfl|rfl|rfl|rfl)|(rfl|rfl|rfl|rfl)
    all_goals simp [ac.queryShape,ac.keyShape,ac.valueShape,ac.incomingShape,ac.outputShape,
      ac.queryGradientShape,ac.keyGradientShape,ac.valueGradientShape,queryShape,valueShape,hb,hh,hn,hD,hE]
  simpa [span,ChunkDeltaFwd.Host.span] using layout_span_list t.shape t.strides hp hd

theorem strides_four (xs : List Nat) (hl : xs.length=4) :
    ∃ sb sh st sf, xs=[sb,sh,st,sf] := by
  match he : xs with
  | [sb,sh,st,sf] => exact ⟨sb,sh,st,sf,rfl⟩
  | [] | [_] | [_,_] | [_,_,_] | _::_::_::_::_::_ => simp [he] at hl

/-- Original logical view coordinates remain inside their storage spans even
when strides overlap or are zero. No contiguity assumption is used. -/
theorem strided_coordinate_bound (d : Dimensions) (src : Tensor) (width : Nat)
    (hs : src.shape=[d.B,d.H,d.N,width]) (hl : src.strides.length=4)
    (b : Fin d.B) (h : Fin d.H) (t : Fin d.N) (f : Fin width) :
    offset4 src b h t f<span src := by
  obtain ⟨sb,sh,st,sf,he⟩ := strides_four src.strides hl
  have hb := Nat.mul_le_mul_right sb (Nat.le_sub_one_of_lt b.isLt)
  have hh := Nat.mul_le_mul_right sh (Nat.le_sub_one_of_lt h.isLt)
  have ht := Nat.mul_le_mul_right st (Nat.le_sub_one_of_lt t.isLt)
  have hf := Nat.mul_le_mul_right sf (Nat.le_sub_one_of_lt f.isLt)
  simp only [offset4,he,List.getElem!_cons_zero,List.getElem!_cons_succ,
    span,ChunkDeltaFwd.Host.span,hs,List.zip_cons_cons,List.zip_nil_left,
    List.map_cons,List.map_nil,List.sum_cons,List.sum_nil]
  omega

theorem layout_coordinate (d : Dimensions) (src : Tensor) (width : Nat)
    (hs : src.shape=[d.B,d.H,d.N,width]) (hl : src.strides.length=4)
    (layout : DenseLayout src.shape src.strides)
    (b : Fin d.B) (h : Fin d.H) (t : Fin d.N) (f : Fin width) :
    offset4 src b h t f=packed d width b h t f := by
  obtain ⟨sb,sh,st,sf,he⟩ := strides_four src.strides hl
  simp only [hs,he,DenseLayout,List.prod_cons,List.prod_nil,Nat.mul_one] at layout
  have hb : b.val*sb=b.val*(d.H*(d.N*width)) := by
    rcases layout.1 with hb|hb
    · have hz : b.val=0 := by have := b.isLt; omega
      simp [hz]
    · rw [hb]
  have hh : h.val*sh=h.val*(d.N*width) := by
    rcases layout.2.1 with hh|hh
    · have hz : h.val=0 := by have := h.isLt; omega
      simp [hz]
    · rw [hh]
  have ht : t.val*st=t.val*width := by
    rcases layout.2.2.1 with ht|ht
    · have hz : t.val=0 := by have := t.isLt; omega
      simp [hz]
    · rw [ht]
  have hf : f.val*sf=f.val := by
    rcases layout.2.2.2.1 with hf|hf
    · have hz : f.val=0 := by have := f.isLt; omega
      simp [hz]
    · simp [hf]
  simp only [offset4,he,List.getElem!_cons_zero,List.getElem!_cons_succ,hb,hh,ht,hf,packed]
  ring

theorem forward_output_survives {α : Type} (c : Invocation) (cc : InvocationContract c)
    (a : Allocations) (ac : AllocationContract c a) (pa : BackwardPhysicalAllocation c a)
    (values : RegionName → Nat → α) (before : Nat → α) (o : Nat) (ho : o<a.output.shape.prod) :
    publishBackward a values before (a.output.base+o)=before (a.output.base+o) := by
  apply backward_published_inputs c a pa values before a.output
    (by simp [backwardProtected]) o
  rwa [allocated_span c cc a ac a.output (by simp [Allocations.outputs])]

theorem packed_query (d : Dimensions) (head time feature : Nat)
    (hh : head<d.B*d.H) (ht : time<d.N) (hf : feature<d.D) :
    head*d.N*d.D+time*d.D+feature<(queryShape d).prod := by
  have hb := Geometry.dense_bound hh ht hf
  simpa [Geometry.dense,queryShape,Nat.add_mul,Nat.mul_assoc] using hb
theorem packed_value (d : Dimensions) (head time feature : Nat)
    (hh : head<d.B*d.H) (ht : time<d.N) (hf : feature<d.E) :
    head*d.N*d.E+time*d.E+feature<(valueShape d).prod := by
  have hb := Geometry.dense_bound hh ht hf
  simpa [Geometry.dense,valueShape,Nat.add_mul,Nat.mul_assoc] using hb

/-- Logical regions outside the eight wrapper descriptors retain their
fallback interpretation, so no unmentioned physical object is fabricated. -/
theorem logical_fallback {α : Type} (a : Allocations) (physical : Nat → Cell α)
    (fallback : Memory α) (r : RegionName) (o : Nat) (hr : descriptor a r=none) :
    logical a physical fallback r o=fallback r o := by simp [logical,hr]
theorem device_pointer_safe (c : Invocation) (a : Allocations) (pa : PhysicalAllocation c a)
    (b : Physical.Buffer) (hb : b∈deviceBuffers a) (offset : Nat) (ho : offset<c.dims.formedLimit) :
    4*(b.base+offset)<2^63 := pointer_safe _ _ pa.pointerRoom b hb offset ho
theorem forward_pointer_safe (c : Invocation) (a : Allocations) (pa : ForwardPhysicalAllocation c a)
    (b : Physical.Buffer) (hb : b∈forwardDeviceBuffers a) (offset : Nat) (ho : offset<c.dims.formedLimit) :
    4*(b.base+offset)<2^63 := pointer_safe _ _ pa.pointerRoom b hb offset ho
theorem backward_pointer_safe (c : Invocation) (a : Allocations) (pa : BackwardPhysicalAllocation c a)
    (b : Physical.Buffer) (hb : b∈backwardDeviceBuffers a) (offset : Nat) (ho : offset<c.dims.formedLimit) :
    4*(b.base+offset)<2^63 := pointer_safe _ _ pa.pointerRoom b hb offset ho

#axiomsClean publication_read
#axiomsClean publication_inputs
#axiomsClean forward_published_read
#axiomsClean forward_published_inputs
#axiomsClean forward_published_frame
#axiomsClean backward_published_read
#axiomsClean backward_published_inputs
#axiomsClean backward_published_frame
#axiomsClean two_stage_frame
#axiomsClean published_read
#axiomsClean published_inputs
#axiomsClean published_frame
#axiomsClean output_regions
#axiomsClean output_sizes
#axiomsClean allocated_span
#axiomsClean strided_coordinate_bound
#axiomsClean layout_coordinate
#axiomsClean forward_output_survives
#axiomsClean packed_query
#axiomsClean packed_value
#axiomsClean logical_fallback
#axiomsClean device_pointer_safe
#axiomsClean forward_pointer_safe
#axiomsClean backward_pointer_safe
#specNonCircular PublicationAllocation avoiding [publish,publishForward,publishBackward]
#specNonCircular PhysicalAllocation avoiding [publish,publishForward,publishBackward]
#specNonCircular ForwardPhysicalAllocation avoiding [publish,publishForward,publishBackward]
#specNonCircular BackwardPhysicalAllocation avoiding [publish,publishForward,publishBackward]
end
end VeriTile.Bench.Repairs.LightningAttention.Host
