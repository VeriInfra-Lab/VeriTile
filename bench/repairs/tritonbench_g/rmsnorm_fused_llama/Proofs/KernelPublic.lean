import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.KernelMemory

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.KernelPublic
open VeriTile Triton FP.Structural Contract KernelAPI KernelMemory
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
set_option maxHeartbeats 4000000
variable {x w out : TensorInfo} {eps : Epsilon}

def Access (t : Torch.FP32.Tensor) (i : Nat) : Prop :=
  t.storage.address (t.physicalIndex i)=t.dataPtr.toNat+4*i ∧
  t.physicalIndex i<t.storage.elements ∧ t.storage.address (t.physicalIndex i)+4≤2^64

def ActualAccesses (b : Binding x w out) (q : Host.Request) : Prop :=
  ∀ row col,row<q.rows → col<q.columns →
    Access b.input (row*q.columns+col) ∧ Access b.weight col ∧ Access b.output (row*q.columns+col)

theorem accesses (b : Binding x w out) (ok : Admitted x w eps) : ActualAccesses b (request x w eps ok) := by
  have h := binding_facts b ok
  intro row col hr hc
  have hi := Index.active_bound (request x w eps ok) row col hr hc
  exact ⟨access b.input b.inputLaws h.1 _ (by rw [h.2.2.2.2.2.1];exact hi),
    access b.weight b.weightLaws h.2.1 _ (by rw [h.2.2.2.2.2.2.1];exact hc),
    access b.output b.outputLaws h.2.2.1 _ (by rw [h.2.2.2.2.2.2.2];exact hi)⟩

theorem alignment (b : Binding x w out) (ok : Admitted x w eps)
    (nonempty : (request x w eps ok).rows≠0) :
    b.input.dataPtr.toNat%16=0 ∧ b.weight.dataPtr.toNat%16=0 := by
  rcases (python_checks_iff x w eps).mp ok with ⟨_,_,_,_,_,_,_,_,_,_,aligned⟩
  have facts := request_facts x w eps ok
  have positive := (request x w eps ok).positive
  have hn : x.numel≠0 := by
    rw [←facts.2.2.2,Host.Request.size]
    exact Nat.mul_ne_zero nonempty (Nat.ne_of_gt positive)
  rcases aligned with empty | aligned
  · exact False.elim (hn empty)
  · simpa [Torch.FP32.Tensor.dataPtr,b.inputMetadata,b.weightMetadata] using aligned

theorem metadata (b : Binding x w out) :
    b.output.metadata.shape=b.input.metadata.shape ∧
    b.output.metadata.dtype=.floating .fp32 ∧ b.output.metadata.device=b.input.metadata.device := by
  refine ⟨?_,b.outputLaws.dtype,?_⟩
  · simpa [b.outputMetadata,b.inputMetadata] using b.outputShape
  · simpa [b.outputMetadata,b.inputMetadata] using b.outputDevice

theorem input_words (b : Binding x w out) (ok : Admitted x w eps) (before : Raw.Memory) :
    (∀ i,(logicalWord before b.input b.inputLaws i).bits=tensorRead (wordRead before) b.input i) ∧
    (∀ i,(logicalWord before b.weight b.weightLaws i).bits=tensorRead (wordRead before) b.weight i) := by
  have h := binding_facts b ok
  exact ⟨logical_word before b.input b.inputLaws h.2.2.2.1,
    logical_word before b.weight b.weightLaws h.2.2.2.2.1⟩

theorem backing_outside (b : Binding x w out) (ok : Admitted x w eps)
    (t : Torch.FP32.Tensor) (sep : Separate t b.output) (i : Nat) (hi : i<t.storage.elements) (byte : Fin 4) :
    Outside (request x w eps ok) b t.metadata.device.index (t.storage.address i+byte.val) := by
  intro j hj overlap
  have f := binding_facts b ok
  have hnum : j<b.output.metadata.numel := by rw [f.2.2.2.2.2.2.2];exact hj
  have bound := b.outputLaws.inBounds j hnum
  rw [b.outputLaws.contiguous f.2.2.1 j hnum] at bound
  rcases overlap with ⟨owner,lo,up⟩
  simp only [key,Torch.FP32.Storage.address] at owner lo up ⊢
  rcases sep with different | before | after
  · exact different owner
  · omega
  · omega

theorem backings_frame (b : Binding x w out) (ok : Admitted x w eps)
    (before after : Raw.Memory)
    (frame : ∀ owner address,Outside (request x w eps ok) b owner address → after owner address=before owner address) :
    ∀ t,t=b.input ∨ t=b.weight → ∀ i,i<t.storage.elements → ∀ byte : Fin 4,
      after t.metadata.device.index (t.storage.address i+byte.val)=before t.metadata.device.index (t.storage.address i+byte.val) := by
  intro t ht i hi byte
  rcases ht with rfl | rfl
  · exact frame _ _ (backing_outside b ok _ b.separateInput i hi byte)
  · exact frame _ _ (backing_outside b ok _ b.separateWeight i hi byte)

theorem real_tensor (b : Binding x w out) (ok : Admitted x w eps) (before : Heap ℝ)
    (values : Nat → ℝ) (i : Nat) (hi : i<(request x w eps ok).size) :
    tensorRead (outputReals (request x w eps ok) b values before) b.output i=values i := by
  have h := binding_facts b ok
  rw [←coordinate _ b.output b.outputLaws h.2.2.1 i (by rw [h.2.2.2.2.2.2.2];exact hi)]
  exact real_read _ b values before i hi

theorem native_tensor (b : Binding x w out) (ok : Admitted x w eps) (before : Raw.Memory)
    (values : Nat → Native.Word) (i : Nat) (hi : i<(request x w eps ok).size) :
    (logicalWord (outputBytes (request x w eps ok) b values before) b.output b.outputLaws i).bits=values i := by
  have h := binding_facts b ok
  rw [logical_word _ b.output b.outputLaws (by simpa [b.outputMetadata] using b.outputNegative)]
  rw [←coordinate _ b.output b.outputLaws h.2.2.1 i (by rw [h.2.2.2.2.2.2.2];exact hi)]
  exact bytes_read _ b values before i hi

/-- Empty leading dimensions launch no source/candidate program and write no
bytes, without an alignment or nonempty-storage premise. -/
theorem empty_kernel (q : Host.Request) (h : q.rows=0) (b : Binding x w out)
    (R : Native.Reductions) (epsilon : Native.Term) (initial : State Native.Term)
    (values : Nat → Native.Word) (before : Raw.Memory) :
    RuntimeGrid.run (Native.algebra R) "x" "y" "w" q.rows q.columns 16384 epsilon initial q.rows=some initial ∧
    RuntimeCandidate.grid (Native.algebra R) "x" "y" "w" q.rows q.columns 16384 epsilon initial q.rows=some initial ∧
    outputBytes q b values before=before := by
  simp [h,RuntimeGrid.run,RuntimeCandidate.grid,outputBytes,Host.Request.size,
    VeriTile.Bench.Repairs.SwigluTriton.Memory.stores]

#axiomsClean accesses
#axiomsClean alignment
#axiomsClean backings_frame
#axiomsClean real_tensor
#axiomsClean native_tensor
#axiomsClean empty_kernel
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.KernelPublic
