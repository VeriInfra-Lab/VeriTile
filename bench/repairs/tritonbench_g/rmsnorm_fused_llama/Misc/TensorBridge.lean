import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.KernelPublic
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.KernelPrecision
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Contract

/-! Optional public Python-check correspondence. -/
namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.TensorBridge
open VeriTile Triton FP.Structural Contract KernelAPI KernelMemory KernelPublic
noncomputable section

theorem python_checks_iff (x w : TensorInfo) (eps : Epsilon) :
    Admitted x w eps ↔ InputContract x w eps := Contract.python_checks_iff x w eps

theorem kernel_correct (x w out : TensorInfo) (eps : Epsilon)
    (c : InputContract x w eps) (b : Binding x w out) (before : Heap ℝ) :
    let ok := (python_checks_iff x w eps).mpr c
    let q := request x w eps ok
    check (.tensor x) (.tensor w) eps=Bool.true ∧
    b.output.metadata.shape=b.input.metadata.shape ∧ b.output.metadata.dtype=.floating .fp32 ∧
    b.output.metadata.device=b.input.metadata.device ∧ ActualAccesses b q ∧
    (q.rows≠0 → b.input.dataPtr.toNat%16=0 ∧ b.weight.dataPtr.toNat%16=0) ∧
    (∀ row lane,row<q.rows → lane<16384 →
      ((BitVec.ofNat 32 row)*(BitVec.ofNat 32 q.columns)+BitVec.ofNat 32 lane).toInt=(row*q.columns+lane:Int)) ∧
    ∃ final finalHeap,
      RealGrid.run "x" "y" "w" q.rows q.columns 16384 q.epsilon (KernelSource.realEntry before b) q.rows=some final ∧
      KernelCandidate.realGrid "x" "y" "w" q.rows q.columns 16384 q.epsilon (KernelSource.realEntry before b) q.rows=some final ∧
      finalHeap=outputReals q b (final.readMem "y") before ∧
      (∀ row col,row<q.rows → col<q.columns →
        tensorRead finalHeap b.output (row*q.columns+col)=
          mathematical (tensorRead before b.input) (tensorRead before b.weight) row q.columns q.epsilon col) ∧
      finalHeap=referenceReals q b before ∧
      (∀ owner address,Outside q b owner address → finalHeap owner address=before owner address) ∧
      RealGrid.Frame "y" q.size (KernelSource.realEntry before b) final := by
  let ok := (python_checks_iff x w eps).mpr c
  let q := request x w eps ok
  obtain ⟨shape,dtype,device⟩ := metadata b
  refine ⟨ok,shape,dtype,device,accesses b ok,alignment b ok,signed_address q,?_⟩
  obtain ⟨final,ran,candidate,values,frame⟩ := KernelSource.real_result before b ok
  refine ⟨final,outputReals q b (final.readMem "y") before,ran,candidate,rfl,?_,real_memory q b before final values,?_,frame⟩
  · intro row col hr hc
    rw [real_tensor b ok before _ _ (Index.active_bound q row col hr hc)]
    exact values row col hr hc
  · exact real_frame q b _ before

theorem kernel_native_equivalence (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (x w out : TensorInfo) (eps : Epsilon) (c : InputContract x w eps)
    (b : Binding x w out) (epsilon : EpsilonABI.Binding eps.value) (before : Raw.Memory) (undefinedWeight : Nat → Native.Word) :
    let ok := (python_checks_iff x w eps).mpr c
    let q := request x w eps ok
    check (.tensor x) (.tensor w) eps=Bool.true ∧
    b.output.metadata.shape=b.input.metadata.shape ∧ b.output.metadata.dtype=.floating .fp32 ∧
    b.output.metadata.device=b.input.metadata.device ∧ ActualAccesses b q ∧
    (q.rows≠0 → b.input.dataPtr.toNat%16=0 ∧ b.weight.dataPtr.toNat%16=0) ∧
    epsilon.runtime=EpsilonABI.convert epsilon.host ∧
    ((scalarABI (fun _=>epsilon.runtime) q).stride.toInt=(q.columns:Int) ∧
      (scalarABI (fun _=>epsilon.runtime) q).columns.toInt=(q.columns:Int) ∧
      (scalarABI (fun _=>epsilon.runtime) q).epsilon=epsilon.runtime) ∧
    ((∀ i,(logicalWord before b.input b.inputLaws i).bits=tensorRead (wordRead before) b.input i) ∧
      (∀ i,(logicalWord before b.weight b.weightLaws i).bits=tensorRead (wordRead before) b.weight i)) ∧
    (∀ row lane,row<q.rows → lane<16384 →
      ((BitVec.ofNat 32 row)*(BitVec.ofNat 32 q.columns)+BitVec.ofNat 32 lane).toInt=(row*q.columns+lane:Int)) ∧
    ∃ final finalBytes,
      RuntimeGrid.run (Native.algebra R) "x" "y" "w" q.rows q.columns 16384 (RuntimeSource.epsilonTerm epsilon.runtime)
        (KernelSource.entry (wordRead before) b undefinedWeight) q.rows=some final ∧
      RuntimeCandidate.grid (Native.algebra R) "x" "y" "w" q.rows q.columns 16384 (RuntimeSource.epsilonTerm epsilon.runtime)
        (KernelSource.entry (wordRead before) b undefinedWeight) q.rows=some final ∧
      finalBytes=outputBytes q b (Native.observe I S final "y") before ∧
      (∀ row col,row<q.rows → col<q.columns →
        (logicalWord finalBytes b.output b.outputLaws (row*q.columns+col)).bits=
          RuntimeSource.nativeTarget I R S (wordRead before) b q epsilon.runtime row col) ∧
      (∀ row col,row<q.rows → col<q.columns → ∀ byte : Fin 4,
        finalBytes (key b.output (row*q.columns+col)).owner ((key b.output (row*q.columns+col)).address+byte.val)=
          Raw.unpack (RuntimeSource.nativeTarget I R S (wordRead before) b q epsilon.runtime row col) byte) ∧
      finalBytes=referenceBytes I R S q b epsilon.runtime before ∧
      (∀ owner address,Outside q b owner address → finalBytes owner address=before owner address) ∧
      (∀ t,t=b.input ∨ t=b.weight → ∀ i,i<t.storage.elements → ∀ byte : Fin 4,
        finalBytes t.metadata.device.index (t.storage.address i+byte.val)=before t.metadata.device.index (t.storage.address i+byte.val)) ∧
      RuntimeGrid.Frame "y" q.size (KernelSource.entry (wordRead before) b undefinedWeight) final := by
  let ok := (python_checks_iff x w eps).mpr c
  let q := request x w eps ok
  obtain ⟨shape,dtype,device⟩ := metadata b
  refine ⟨ok,shape,dtype,device,accesses b ok,alignment b ok,
    epsilon.converted,scalar_abi (fun _=>epsilon.runtime) q,input_words b ok before,signed_address q,?_⟩
  obtain ⟨final,ran,candidate,values,frame⟩ := RuntimeSource.native_result I R S (wordRead before) b ok epsilon.runtime undefinedWeight
  have exterior := bytes_frame q b (Native.observe I S final "y") before
  refine ⟨final,outputBytes q b (Native.observe I S final "y") before,ran,candidate,rfl,?_,?_,
    native_memory I R S q b epsilon.runtime before final values,exterior,backings_frame b ok before _ exterior,frame⟩
  · intro row col hr hc
    rw [native_tensor b ok before _ _ (Index.active_bound q row col hr hc)]
    exact values row col hr hc
  · intro row col hr hc byte
    rw [bytes_value q b _ before _ (Index.active_bound q row col hr hc) byte,values row col hr hc]

#axiomsClean kernel_correct
#axiomsClean kernel_native_equivalence
#axiomsClean python_checks_iff
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.TensorBridge
