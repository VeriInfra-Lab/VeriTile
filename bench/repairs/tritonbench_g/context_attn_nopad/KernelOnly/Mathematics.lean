import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.FPValues

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 3000000

def tensorIndex (c : Call) (row head dim : Nat) : Nat := (row*c.q.heads+head)*c.q.dims+dim

/-- Independent logical Tensor score, using the actual Python binary64 scale. -/
def mathematicalScore (r : Request) (read : Nat → ℝ) (b head t j : Nat) : ℝ :=
  (∑ d : Fin r.call.q.dims,
    logical read r.q (tensorIndex r.call (r.call.start b+t) head d.val) *
    logical read r.k (tensorIndex r.call (r.call.start b+j) head d.val)) * r.call.scale

/-- Self-inclusive causal attention: exactly keys 0 ≤ j ≤ t. This target has
no PID, padded lane, online recurrence, native evaluator or output memory. -/
def mathematical (r : Request) (read : Nat → ℝ) (b head t d : Nat) : ℝ :=
  (∑ j ∈ Finset.range (t+1),Real.exp (mathematicalScore r read b head t j) *
    logical read r.v (tensorIndex r.call (r.call.start b+j) head d)) /
  ∑ j ∈ Finset.range (t+1),Real.exp (mathematicalScore r read b head t j)

/-- Unrestricted Real interpretation of the same byte-addressed Tensor
descriptors. Metadata is read from the actual original INT32 byte heap;
payload Real values need no floating encoder or finiteness hypothesis. -/
def realInitial (r : Request) (heap : ByteMemory) (read : Nat → ℝ) : BlockState where
  mem := fun rg off => if rg=regions.start then ⟨.int,startValues r heap off⟩
    else if rg=regions.sequence then ⟨.int,lengthValues r heap off⟩
    else .real (read (base r rg+4*off))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := r.call.grid.toNumPids

def realObservation (r : Request) (heap : ByteMemory) (read : Nat → ℝ) :
    Checked.MetadataObservation r.call regions (realInitial r heap read)
      (startValues r heap) (lengthValues r heap) where
  startWords := fun b => load32 heap (r.start.metadata.dataPtr.toNat+4*b)
  lengthWords := fun b => load32 heap (r.length.metadata.dataPtr.toNat+4*b)
  startCPU := fun _ _ => rfl
  lengthCPU := fun _ _ => rfl
  startMemory := by intro b hb; rfl
  lengthMemory := by intro b hb; rfl

theorem real_memory (r : Request) (heap : ByteMemory) (h : Contract r heap) (read : Nat → ℝ) :
    Host.Memory r.call regions (allocation r) (realInitial r heap read) :=
  Checked.memory r.call regions (allocation r) (realInitial r heap read)
    (startValues r heap) (lengthValues r heap) h.checks.2.1 h.checks.2.2.2
    (realObservation r heap read) labels

theorem tensor_index_bound (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (row head dim : Nat) (hr : row<r.call.q.rows) (hh : head<r.call.q.heads) (hd : dim<r.call.q.dims) :
    tensorIndex r.call row head dim<payloadCount r := by
  have canonical := h.checks.2.1.2.2.2.2.2.1
  have bounded := Checked.bounded r.call.q canonical (payloadCount r) (le_refl _) row head dim hr hh hd
  simpa only [Checked.offset_eq r.call.q canonical,tensorIndex] using bounded

theorem real_leaf (r : Request) (heap : ByteMemory) (h : Contract r heap) (read : Nat → ℝ)
    (role : InputRole) (row head dim : Nat)
    (hr : row<r.call.q.rows) (hh : head<r.call.q.heads) (hd : dim<r.call.q.dims) :
    (realInitial r heap read).readMem role.region (r.call.q.offset row head dim)=
      logical read (role.payload r) (tensorIndex r.call row head dim) := by
  have member : (role.payload r).view∈[(views r).q,(views r).k,(views r).v,(views r).o] := by
    cases role <;> simp [InputRole.payload,views]
  have shape : (role.payload r).tensor.metadata.shape=[r.call.q.rows,r.call.q.heads,r.call.q.dims] :=
    (h.checks.1.1 _ member).1
  have bound : tensorIndex r.call row head dim<(role.payload r).tensor.metadata.numel := by
    simpa [Torch.TensorMetadata.numel,shape,payloadCount,Nat.mul_assoc] using
      tensor_index_bound r heap h row head dim hr hh hd
  rw [logical_direct read _ (inputAPI r heap h role) (input_descriptor r heap h role).1 _ bound,
    Checked.offset_eq r.call.q h.checks.2.1.2.2.2.2.2.1]
  cases role <;> rfl

theorem score_eq (r : Request) (heap : ByteMemory) (h : Contract r heap) (read : Nat → ℝ)
    (b head t j : Nat) (hb : b<r.call.batches) (hh : head<r.call.q.heads)
    (ht : t<r.call.length b) (hj : j≤t) :
    r.call.score (Program.inputs (realInitial r heap read) regions) b head t j=
      mathematicalScore r read b head t j := by
  have capacity := ((real_memory r heap h read).shape.qRows b hb)
  have qt : r.call.start b+t<r.call.q.rows := by omega
  have kj : r.call.start b+j<r.call.q.rows := by omega
  simp only [Call.score,mathematicalScore,Program.inputs,h.checks.2.1.2.2.1]
  congr 1
  apply Finset.sum_congr rfl
  intro d _
  exact congrArg₂ (· * ·) (real_leaf r heap h read .query _ head d.val qt hh d.isLt)
    (real_leaf r heap h read .key _ head d.val kj hh d.isLt)

theorem expected_eq (r : Request) (heap : ByteMemory) (h : Contract r heap) (read : Nat → ℝ)
    (b head t d : Nat) (hb : b<r.call.batches) (hh : head<r.call.q.heads)
    (ht : t<r.call.length b) (hd : d<r.call.q.dims) :
    r.call.expected (Program.inputs (realInitial r heap read) regions) b head t d=
      mathematical r read b head t d := by
  have capacity := ((real_memory r heap h read).shape.qRows b hb)
  unfold Call.expected mathematical
  congr 1
  · apply Finset.sum_congr rfl
    intro j hj
    have jt : j≤t := by have := Finset.mem_range.mp hj; omega
    rw [score_eq r heap h read b head t j hb hh ht jt]
    congr 1
    simp only [Program.inputs,h.checks.2.1.2.2.2.1]
    exact real_leaf r heap h read .value _ head d (by omega) hh hd
  · apply Finset.sum_congr rfl
    intro j hj
    rw [score_eq r heap h read b head t j hb hh ht (by have := Finset.mem_range.mp hj; omega)]

#specNonCircular mathematical avoiding [VeriTile.Triton.exec,FP.Structural.exec,FPReference.grid,execute,nativeStores]
#axiomsClean expected_eq
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
