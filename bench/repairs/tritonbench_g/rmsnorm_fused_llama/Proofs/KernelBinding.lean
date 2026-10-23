import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Contract
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.Index
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.KernelAPI
open VeriTile Torch Contract
open scoped BigOperators
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000

abbrev Heap (α : Type) := Nat → Nat → α

/-- Logical element reads use the actual Tensor's storage and physical index. -/
def tensorRead (read : Heap α) (t : FP32.Tensor) (i : Nat) : α :=
  read t.metadata.device.index (t.storage.address (t.physicalIndex i))

def sourceRead (read : Heap α) (t : FP32.Tensor) (i : Nat) : α :=
  read t.metadata.device.index (t.storage.address (t.storageOffset+i))

def mathematical (xs ws : Nat → ℝ) (row n : Nat) (eps : ℝ) (k : Nat) : ℝ :=
  xs (row*n+k) * (Real.sqrt ((∑ j : Fin n,xs (row*n+j.val)^2)/(n : ℝ)+eps))⁻¹ * ws k

/-- Fresh output allocation is disjoint from each entire original backing.
Input and weight are allowed to be the same object or overlapping views. -/
def Separate (x y : FP32.Tensor) : Prop :=
  x.metadata.device.index≠y.metadata.device.index ∨
  x.storage.base+4*x.storage.elements≤y.storage.base ∨
  y.storage.base+4*y.storage.elements≤x.storage.base

/-- Ordinary Tensor/empty_like facts only. No state, interpreter, graph,
execution outcome or numerical result occurs in these API bindings. -/
structure Binding (x w out : TensorInfo) where
  input : FP32.Tensor
  weight : FP32.Tensor
  output : FP32.Tensor
  inputLaws : FP32.TensorLaws input
  weightLaws : FP32.TensorLaws weight
  outputLaws : FP32.TensorLaws output
  inputMetadata : input.metadata=x.toTensorMetadata
  weightMetadata : weight.metadata=w.toTensorMetadata
  outputMetadata : output.metadata=out.toTensorMetadata
  inputObject : input.object=x.object
  weightObject : weight.object=w.object
  outputObject : output.object=out.object
  inputBacking : input.storage.object=x.backing
  weightBacking : weight.storage.object=w.backing
  outputBacking : output.storage.object=out.backing
  outputShape : out.shape=x.shape
  outputDevice : out.device=x.device
  outputContiguous : out.isContiguous=Bool.true
  outputNegative : out.isNeg=Bool.false
  outputConjugate : out.isConj=Bool.false
  freshInput : input.storage.object≠output.storage.object
  freshWeight : weight.storage.object≠output.storage.object
  separateInput : Separate input output
  separateWeight : Separate weight output

variable {x w out : TensorInfo} {eps : Epsilon}

theorem coordinate (read : Heap α) (t : FP32.Tensor) (api : FP32.TensorLaws t)
    (contiguous : t.metadata.isContiguous=Bool.true) (i : Nat) (hi : i<t.metadata.numel) :
    sourceRead read t i=tensorRead read t i := by
  simp only [sourceRead,tensorRead,api.contiguous contiguous i hi]

theorem pointer (t : FP32.Tensor) (api : FP32.TensorLaws t)
    (contiguous : t.metadata.isContiguous=Bool.true) (nonempty : 0<t.metadata.numel) :
    t.dataPtr.toNat=t.storage.address t.storageOffset := by
  have cap := FP32.contiguous_capacity t api contiguous nonempty
  have span := t.storage.spanFits
  have fits : t.storage.address t.storageOffset<2^64 := by
    unfold FP32.Storage.address
    omega
  rw [api.pointer nonempty]
  simp only [BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt fits

theorem access (t : FP32.Tensor) (api : FP32.TensorLaws t)
    (contiguous : t.metadata.isContiguous=Bool.true) (i : Nat) (hi : i<t.metadata.numel) :
    t.storage.address (t.physicalIndex i)=t.dataPtr.toNat+4*i ∧
    t.physicalIndex i<t.storage.elements ∧ t.storage.address (t.physicalIndex i)+4≤2^64 := by
  have eq := pointer t api contiguous (by omega)
  have bounds := api.inBounds i hi
  have span := t.storage.spanFits
  refine ⟨?_,bounds,?_⟩
  · rw [api.contiguous contiguous i hi,eq]
    simp [FP32.Storage.address,Nat.mul_add,Nat.add_assoc]
  · unfold FP32.Storage.address
    omega

theorem binding_facts (b : Binding x w out) (ok : Admitted x w eps) :
    b.input.metadata.isContiguous=Bool.true ∧ b.weight.metadata.isContiguous=Bool.true ∧
    b.output.metadata.isContiguous=Bool.true ∧
    b.input.metadata.isNeg=Bool.false ∧ b.weight.metadata.isNeg=Bool.false ∧
    b.input.metadata.numel=(request x w eps ok).size ∧
    b.weight.metadata.numel=(request x w eps ok).columns ∧
    b.output.metadata.numel=(request x w eps ok).size := by
  rcases (python_checks_iff x w eps).mp ok with ⟨_,_,_,hc,hn,_,_,hw,_,_,_⟩
  have req := request_facts x w eps ok
  simp only [b.inputMetadata,b.weightMetadata,b.outputMetadata]
  exact ⟨hc.1,hc.2,b.outputContiguous,hn.1.1,hn.2,req.2.2.2.symm,
    hw.2.2.trans req.2.1.symm,by simpa [Torch.TensorMetadata.numel,b.outputShape] using req.2.2.2.symm⟩

theorem input_read (b : Binding x w out) (ok : Admitted x w eps) (read : Heap α)
    (row col : Nat) (hr : row<(request x w eps ok).rows) (hc : col<(request x w eps ok).columns) :
    sourceRead read b.input (row*(request x w eps ok).columns+col)=
      tensorRead read b.input (row*(request x w eps ok).columns+col) := by
  have h := binding_facts b ok
  exact coordinate read b.input b.inputLaws h.1 _ (by rw [h.2.2.2.2.2.1];exact Index.active_bound _ _ _ hr hc)

theorem weight_read (b : Binding x w out) (ok : Admitted x w eps) (read : Heap α)
    (col : Nat) (hc : col<(request x w eps ok).columns) :
    sourceRead read b.weight col=tensorRead read b.weight col := by
  have h := binding_facts b ok
  exact coordinate read b.weight b.weightLaws h.2.1 _ (by rw [h.2.2.2.2.2.2.1];exact hc)

/-- ABI values for the two signed-i32 runtime integers and FP32 epsilon.
`encode` is the scalar host-float-to-FP32 conversion, never a row callback. -/
structure ScalarABI where
  stride : BitVec 32
  columns : BitVec 32
  epsilon : BitVec 32

def scalarABI (encode : ℝ → BitVec 32) (q : Host.Request) : ScalarABI :=
  ⟨BitVec.ofNat 32 q.columns,BitVec.ofNat 32 q.columns,encode q.epsilon⟩

theorem scalar_abi (encode : ℝ → BitVec 32) (q : Host.Request) :
    (scalarABI encode q).stride.toInt=(q.columns : Int) ∧
    (scalarABI encode q).columns.toInt=(q.columns : Int) ∧
    (scalarABI encode q).epsilon=encode q.epsilon := by
  have h : (BitVec.ofNat 32 q.columns).toInt=(q.columns : Int) := by
    have hn : (BitVec.ofNat 32 q.columns).toNat=q.columns := by
      simp only [BitVec.toNat_ofNat]
      apply Nat.mod_eq_of_lt
      have := q.limit
      omega
    rw [BitVec.toInt_eq_toNat_of_lt,hn]
    · rw [hn]
      have := q.limit
      omega
  exact ⟨h,h,rfl⟩

theorem signed_address (q : Host.Request) (row lane : Nat) (hr : row<q.rows) (hl : lane<16384) :
    ((BitVec.ofNat 32 row)*(BitVec.ofNat 32 q.columns)+BitVec.ofNat 32 lane).toInt=
      (row*q.columns+lane : Int) := by
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add]
  have bound := Index.padded_index_bound q row lane hr hl
  have hn : (BitVec.ofNat 32 (row*q.columns+lane)).toNat=row*q.columns+lane := by
    simp only [BitVec.toNat_ofNat]
    apply Nat.mod_eq_of_lt
    omega
  rw [BitVec.toInt_eq_toNat_of_lt,hn]
  · push_cast; rfl
  · rw [hn];omega

#axiomsClean binding_facts
#axiomsClean input_read
#axiomsClean weight_read
#axiomsClean access
#axiomsClean scalar_abi
#axiomsClean signed_address
#guard_msgs (drop info) in
#specNonCircular mathematical avoiding [kernel,Row.value,Grid.run,RealGrid.run]
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.KernelAPI
