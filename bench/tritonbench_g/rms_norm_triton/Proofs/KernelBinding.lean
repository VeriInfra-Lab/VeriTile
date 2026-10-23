import bench.tritonbench_g.rms_norm_triton.Proofs.KernelBoundary

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
variable {x w xc wc y : TensorView}

 theorem TensorBinding.numel (h : TensorBinding t v) : t.numel = v.numel := by
  simp only [Tensor.numel,TensorView.numel,h.shape]

 theorem TensorBinding.read_eq (h : TensorBinding t v) (memory : Memory α)
    (i : Nat) (hi : i < t.numel) : t.read memory i = v.read memory i := by
  simp only [Tensor.read,TensorView.read,h.allocation,h.offsets i hi]

 theorem TensorBinding.contiguous_offset (h : TensorBinding t v) (hc : Contiguous t)
    (i : Nat) (hi : i < v.numel) : v.offset i = v.storageOffset+i := by
  have bound : i < t.numel := by rw [h.numel]; exact hi
  rw [← h.offsets i bound,hc i bound,h.base]

/-- Actual byte-address equality for a copied input or output, including the
queried nonzero storage offset. Original strided views are read through their
own `TensorView.offset`; no contiguity premise is imposed on them. -/
 theorem TensorBinding.contiguous_address (h : TensorBinding t v) (hc : Contiguous t)
    (i : Nat) (hi : i < v.numel) :
    v.address i = v.metadata.dataPtr.toNat+4*i := by
  rw [TensorView.address,h.contiguous_offset hc i hi,h.pointer (Nat.zero_lt_of_lt hi)]
  omega

 theorem Prepared.output_numel (a : Prepared α input weight entry preexisting)
    (h : TensorBindings a x w xc wc y) : y.numel = input.numel := by
  simp only [← h.output.numel,Tensor.numel,a.output.shape,a.xc.shape]

 theorem Prepared.output_offset (a : Prepared α input weight entry preexisting)
    (h : TensorBindings a x w xc wc y) (i : Nat) (hi : i < y.numel) :
    y.offset i = a.output.tensor.base+i := by
  rw [h.output.contiguous_offset a.output.contiguous i hi,h.output.base]

 theorem kernel_pointers (a : Prepared α input weight entry preexisting)
    (h : TensorBindings a x w xc wc y) : KernelPointers xc wc y :=
  ⟨h.inputCopy.contiguous_address a.xc.contiguous,
    h.weightCopy.contiguous_address a.wc.contiguous,
    h.output.contiguous_address a.output.contiguous⟩

/-- All flat logical elements are covered by a successful source grid. The
copy law binds each loaded value to its original strided/broadcast Tensor. -/
 theorem prepared_tensor_result [Inhabited α] (a : Prepared α input weight entry preexisting)
    (mdl : Algebra α) (normalized : List Nat) (eps : ℝ)
    (domain : InputDomain input weight normalized preexisting)
    (h : TensorBindings a x w xc wc y) (schedule : Schedule input normalized) :
    ∃ final, a.entryRun mdl normalized eps schedule = some final ∧
      KernelResult (fun k => rowValue mdl (a.layout normalized) eps
        (fun j => x.read entry ((k / columns normalized)*columns normalized+j))
        (w.read entry) (k % columns normalized)) y a.output.memory (a.commit final) := by
  obtain ⟨final,ran,values,frame,_⟩ := a.entry_result mdl normalized eps domain schedule
  have shape := shape_product domain
  refine ⟨final,ran,?_,?_⟩
  · intro k hk
    have kn : k < input.numel := by rw [← a.output_numel h]; exact hk
    have positive : 0 < columns normalized := by
      by_contra z
      have zero : columns normalized = 0 := by omega
      rw [zero,Nat.mul_zero] at shape
      omega
    have rb : k / columns normalized < rows input normalized :=
      (Nat.div_lt_iff_lt_mul positive).2 (by omega)
    have cb := Nat.mod_lt k positive
    have flat : k/columns normalized*columns normalized+k%columns normalized = k := by
      simpa [Nat.mul_comm] using Nat.div_add_mod k (columns normalized)
    have outBound : k < a.output.tensor.numel := by rw [h.output.numel]; exact hk
    rw [← h.output.read_eq (a.commit final) k outBound,← flat]
    rw [values ⟨_,rb⟩ ⟨_,cb⟩]
    simp only [Fin.val_mk,flat]
    apply rowValue_congr _ _ _ _ _ _ _ _ cb
    · intro j hj
      apply h.input.read_eq
      rw [shape]
      calc
        _ < (k/columns normalized)*columns normalized+columns normalized := Nat.add_lt_add_left hj _
        _ = (k/columns normalized+1)*columns normalized := by ring
        _ ≤ rows input normalized*columns normalized := Nat.mul_le_mul_right _ rb
    · intro j hj
      apply h.weight.read_eq
      simpa [Tensor.numel,domain.weightShape,columns] using hj
  · intro allocation offset outside
    apply frame
    intro i hi equal
    apply outside i (by rw [a.output_numel h]; exact hi)
    rw [← h.output.allocation,a.output_offset h i (by rw [a.output_numel h]; exact hi)]
    exact equal

 theorem kernelResult_memory (h : TensorBinding t y) (contiguous : Contiguous t)
    (result : KernelResult formula y initial final) :
    final = referenceMemory formula y initial := by
  funext allocation offset
  unfold referenceMemory
  split
  · rename_i inside
    rcases inside with ⟨same,lower,upper⟩
    have hi : offset-y.storageOffset < y.numel := by omega
    have got := result.1 (offset-y.storageOffset) hi
    simp only [TensorView.read,h.contiguous_offset contiguous _ hi] at got
    simpa only [same,Nat.add_sub_of_le lower] using got
  · rename_i outside
    apply result.2
    intro i hi equal
    apply outside
    have ha := congrArg Prod.fst equal
    have ho := congrArg Prod.snd equal
    simp only [Prod.fst,Prod.snd,h.contiguous_offset contiguous i hi] at ha ho
    exact ⟨ha,by omega,by omega⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
