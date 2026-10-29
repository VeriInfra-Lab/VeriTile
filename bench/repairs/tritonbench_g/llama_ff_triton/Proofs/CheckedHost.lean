import bench.repairs.tritonbench_g.llama_ff_triton.Contract

namespace VeriTile.Bench.Repairs.LlamaFF.CheckedHost
open Contract
noncomputable section
open Classical
set_option maxHeartbeats 5000000

def buffers (x w v r : TensorInfo) (out : Nat) : Buffers :=
  ⟨⟨x.backing,0,0,columns x,1⟩,⟨r.backing,0,1⟩,
   ⟨w.backing,0,columns x,1⟩,⟨v.backing,0,columns x,1⟩,
   ⟨out,0,0,outputs w,1⟩,useByte w⟩
def parameters (x : TensorInfo) : Parameters (BitVec 32) := ⟨columns x,BitVec.ofNat 32 0x358637bd⟩

def descriptor (t : TensorInfo) (dtype : DType) (extents : Nat → Nat) : Tensor :=
  ⟨t.backing,t.storageOffset,t.shape,t.strides,dtype,t.device.index,extents t.backing⟩

/-- Ordinary zero-offset contiguous tensor storage. The capacity and pointer
observations are input/allocator facts; no source execution result appears here. -/
structure TensorStorage (t : TensorInfo) (width : Nat) (extents base bytes : Nat → Nat) : Prop where
  elements : t.numel≤extents t.backing
  pointer : t.dataPtr.toNat=base t.backing
  allocation : width*extents t.backing≤bytes t.backing
  addressSpace : base t.backing+bytes t.backing≤2^64

/-- Actual alias/allocator metadata for the three views and the returned view. -/
structure Binding (x w v r : TensorInfo) where
  extents : Nat → Nat
  base : Nat → Nat
  bytes : Nat → Nat
  input : TensorStorage x 2 extents base bytes
  weight1 : TensorStorage w (weightBytes w) extents base bytes
  weight3 : TensorStorage v (weightBytes w) extents base bytes
  rms : TensorStorage r (weightBytes w) extents base bytes
  allocated : TensorInfo
  output : TensorInfo
  allocatedShape : allocated.shape=[(shape x w).M,(shape x w).N]
  outputShape : output.shape=[batch x,sequence x,outputs w]
  outputType : output.dtype=.other "float16"
  outputDevice : output.device=x.device
  outputDense : Dense output
  outputAlias : output.backing=allocated.backing ∧ output.dataPtr=allocated.dataPtr
  outputStorage : TensorStorage output 2 extents base bytes
  fresh : x.backing≠output.backing ∧ r.backing≠output.backing ∧
    w.backing≠output.backing ∧ v.backing≠output.backing
  flattened : TensorInfo
  flatShape : flattened.shape=[(shape x w).M,(shape x w).K]
  flatAlias : flattened.backing=x.backing ∧ flattened.dataPtr=x.dataPtr
  transposed1 : TensorInfo
  transposed3 : TensorInfo
  transpose1Shape : transposed1.shape=[columns x,outputs w]
  transpose3Shape : transposed3.shape=[columns x,outputs w]
  transpose1Alias : transposed1.backing=w.backing ∧ transposed1.dataPtr=w.dataPtr
  transpose3Alias : transposed3.backing=v.backing ∧ transposed3.dataPtr=v.dataPtr

theorem active_span (h : TensorStorage t width extents base bytes)
    (i byte : Nat) (hi : i<t.numel) (hb : byte<width) :
    t.dataPtr.toNat+width*i+byte<base t.backing+bytes t.backing ∧
    t.dataPtr.toNat+width*i+byte<2^64 := by
  have cap:=h.elements
  have elem : i+1≤extents t.backing := by omega
  have mul:=Nat.mul_le_mul_left width elem
  rw [h.pointer]
  constructor <;> nlinarith [h.allocation,h.addressSpace]

theorem index_domain (ok : InputContract x w v r) (out : Nat) :
    IndexDomain (shape x w) (buffers x w v r out) := by
  have hd:=ok.dimensions
  rcases hd with ⟨hm,hn,hk⟩
  rcases ok.formed with ⟨hx,hw,ho⟩
  have positive : 0<weightBytes w := by unfold weightBytes; split <;> decide
  have fk : formedK (shape x w)<2^63 := by nlinarith
  refine ⟨⟨by norm_num [shape],hm,hn,hk⟩,?_,?_,?_,?_,?_,ok.grid,?_,?_,?_,?_,?_,?_⟩
  · simpa [buffers,shape] using And.intro (by norm_num : 0<2^31) (And.intro hk (by norm_num : 1<2^31))
  · norm_num [buffers]
  · simpa [buffers,shape] using And.intro hk (by norm_num : 1<2^31)
  · simpa [buffers,shape] using And.intro hk (by norm_num : 1<2^31)
  · simpa [buffers,shape] using And.intro (by norm_num : 0<2^31) (And.intro hn (by norm_num : 1<2^31))
  · refine ⟨?_,?_,fk⟩
    · unfold tilesM; omega
    · unfold tilesN; omega
  · simpa [View3.FormedSafe,buffers,shape] using And.intro (by decide : 0<2) hx
  · have bound : weightBytes w*formedK (shape x w)<2^63 :=
      (Nat.mul_le_mul_left (weightBytes w) (Nat.le_add_left (formedK (shape x w)) ((shape x w).N*(shape x w).K))).trans_lt hw
    simpa [View3.FormedSafe,rmsView,buffers,weightBytes] using And.intro positive bound
  · simpa [View3.FormedSafe,weightView,buffers,shape,weightBytes] using And.intro positive hw
  · simpa [View3.FormedSafe,weightView,buffers,shape,weightBytes] using And.intro positive hw
  · simpa [View3.FormedSafe,buffers,shape] using And.intro (by decide : 0<2) ho

theorem dense_index (row col rows cols : Nat) (hr : row<rows) (hc : col<cols) :
    row*cols+col<rows*cols := by nlinarith

theorem output_injective (x w v r : TensorInfo) (out : Nat) :
    Function.Injective (outputAddress (buffers x w v r out) : Cell (shape x w) → Address) := by
  rintro ⟨b,m,n⟩ ⟨b',m',n'⟩ eq
  have hn : n.val<outputs w := n.isLt
  have hn' : n'.val<outputs w := n'.isLt
  have pos : 0<outputs w := by change n.val<outputs w at hn; omega
  have he:=congrArg Prod.snd eq
  simp only [outputAddress,buffers,View3.address,Nat.mul_zero,Nat.zero_add,Nat.mul_one] at he
  have nr : n.val=n'.val := by
    have h:=congrArg (fun z => z%outputs w) he
    simpa [Nat.add_mod,Nat.mod_eq_of_lt hn,Nat.mod_eq_of_lt hn'] using h
  have mr : m.val=m'.val := by
    have h:=congrArg (fun z => z/outputs w) he
    simpa [Nat.mul_comm m.val,Nat.mul_comm m'.val,Nat.mul_add_div pos,
      Nat.div_eq_of_lt hn,Nat.div_eq_of_lt hn'] using h
  have br : b=b' := by
    apply Fin.ext
    have hb : b.val<1 := b.isLt
    have hb' : b'.val<1 := b'.isLt
    omega
  simp only [Prod.mk.injEq]
  exact ⟨br,Fin.ext mr,Fin.ext nr⟩

def domain (x w v r : TensorInfo) (admitted : Admitted x w v r) (a : Binding x w v r) :
    HostDomain (shape x w) (parameters x) (buffers x w v r a.output.backing) := by
  have ok:=(python_checks_iff x w v r).mp admitted
  have xs:=x_shape ok
  have xcap : (shape x w).M*(shape x w).K≤a.extents x.backing := by
    simpa [VeriTile.Torch.TensorMetadata.numel,xs,shape,Nat.mul_assoc] using a.input.elements
  have wcap : (shape x w).N*(shape x w).K≤a.extents w.backing := by
    simpa [VeriTile.Torch.TensorMetadata.numel,ok.shapes.2.2.1,shape] using a.weight1.elements
  have vcap : (shape x w).N*(shape x w).K≤a.extents v.backing := by
    simpa [VeriTile.Torch.TensorMetadata.numel,ok.shapes.2.2.2.1,shape] using a.weight3.elements
  have rcap : (shape x w).K≤a.extents r.backing := by
    simpa [VeriTile.Torch.TensorMetadata.numel,ok.shapes.2.2.2.2,shape] using a.rms.elements
  have ocap : (shape x w).M*(shape x w).N≤a.extents a.output.backing := by
    simpa [VeriTile.Torch.TensorMetadata.numel,a.outputShape,shape,Nat.mul_assoc] using a.outputStorage.elements
  refine {
    valid := ⟨rfl,ok.shapes.1,ok.shapes.2.1⟩
    parameterK := rfl
    epsilon := rfl
    batch := batch x
    seq := sequence x
    flatten := rfl
    device := x.device.index
    extents := a.extents
    input := descriptor x .fp16 a.extents
    w1 := descriptor w (weightDType (buffers x w v r a.output.backing)) a.extents
    w3 := descriptor v (weightDType (buffers x w v r a.output.backing)) a.extents
    rms := descriptor r (weightDType (buffers x w v r a.output.backing)) a.extents
    inputMetadata := ?_
    w1Metadata := ?_
    w3Metadata := ?_
    rmsMetadata := ?_
    xDense := ⟨rfl,rfl,rfl⟩
    w1Dense := ⟨rfl,rfl,rfl⟩
    w3Dense := ⟨rfl,rfl,rfl⟩
    rmsDense := ⟨rfl,rfl⟩
    outputDense := ⟨rfl,rfl,rfl⟩
    fresh := a.fresh
    nonoverlap := output_injective x w v r a.output.backing
    indices := fun _ => index_domain ok a.output.backing
    xAllocated := ?_
    rmsAllocated := ?_
    w1Allocated := ?_
    w3Allocated := ?_
    outputAllocated := ?_ }
  · simp [descriptor,buffers,shape,ok.dense.1.2.1,xs]
  · simp [descriptor,buffers,shape,ok.dense.2.1.2.1,ok.shapes.2.2.1,ok.devices.2.1]
  · simp [descriptor,buffers,shape,ok.dense.2.2.1.2.1,ok.shapes.2.2.2.1,ok.devices.2.2.1]
  · simp [descriptor,buffers,shape,ok.dense.2.2.2.2.1,ok.shapes.2.2.2.2,ok.devices.2.2.2]
  · intro row k hr hk
    simpa [buffers,View3.address,shape] using (dense_index row k _ _ hr hk).trans_le xcap
  · intro k hk
    simpa [buffers,View1.address] using hk.trans_le rcap
  · intro n k hn hk
    simpa [buffers,View2.address,shape] using (dense_index n k _ _ hn hk).trans_le wcap
  · intro n k hn hk
    simpa [buffers,View2.address,shape] using (dense_index n k _ _ hn hk).trans_le vcap
  · intro cell
    simpa [outputAddress,buffers,View3.address,shape] using
      (dense_index cell.2.1.val cell.2.2.val _ _ cell.2.1.isLt cell.2.2.isLt).trans_le ocap

#axiomsClean active_span
#axiomsClean index_domain
#axiomsClean output_injective
#axiomsClean domain
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LlamaFF.CheckedHost
