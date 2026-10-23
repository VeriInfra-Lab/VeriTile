import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.KernelModel
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.KernelCore

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.KernelResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open KernelDomain KernelBinding KernelModel
noncomputable section
local instance : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩
local instance : Inhabited SourceGraph.Expr := ⟨SourceGraph.Expr.zero⟩

private theorem inputs_separate (a : Arguments) : SourceGraph.InputsSeparate (parameters a) := by
  change ("a" : RegionName)≠"out" ∧ ("b" : RegionName)≠"out"
  decide

/-- Successful execution of every actual source program, connected to the
finite-int64 loop/index interpreter. The reference reads initial operands only. -/
theorem graph_correct (rest : Algebra SourceGraph.Expr) (plans : SourceGraph.PlanSelector)
    (a : Arguments) (h : KernelContract a) (native : LaunchABI a)
    (x : Allocated [a.M,a.K]) (y : Allocated [a.M,a.N,a.K]) (z : Allocated [a.M,a.N])
    (heap : Heap) (scratch : State ValueProofs.Word) :
    ∃ final, SourceGraph.runGrid rest plans (parameters a)
        (initial (x.value heap) (y.value heap) scratch)=some final ∧
      FiniteExecution.grid plans (parameters a) (initial (x.value heap) (y.value heap) scratch)=some final.mem ∧
      (∀ m : Fin a.M,∀ n : Fin a.N,final.mem "out" (m.val*a.N+n.val)=
        .mk .real (SourceGraph.tensorValue plans (parameters a)
          (SourceGraph.Native.liftMemory (operandMemory (x.value heap) (y.value heap)))
          m.val n.val (a.K/a.BK))) ∧
      (∀ r i,SourceGraph.GridOutside (parameters a) r i →
        final.mem r i=(initial (x.value heap) (y.value heap) scratch).mem r i) := by
  obtain ⟨final,he,hfinite,hv,hf⟩ := KernelCore.graph_complete rest plans (parameters a)
    (initial (x.value heap) (y.value heap) scratch) (integerAgreement a native x y z)
    (inputs_separate a) h.BM h.BN h.M h.N
  refine ⟨final,he,hfinite,?_,hf⟩
  intro m n
  exact (hv m n).trans
    (congrArg (Cell.mk .real) (tensor_initial plans a (x.value heap) (y.value heap)
      scratch m.val n.val (a.K/a.BK)))

theorem native_correct (rest : Algebra SourceGraph.Expr) (cfg : FP.Config)
    (plans : SourceGraph.PlanSelector) (a : Arguments) (h : KernelContract a) (native : LaunchABI a)
    (x : Allocated [a.M,a.K]) (y : Allocated [a.M,a.N,a.K]) (z : Allocated [a.M,a.N])
    (binding : Binding a x y z) (heap : Heap) (scratch : State ValueProofs.Word) :
    ∃ final, SourceGraph.runGrid rest plans (parameters a)
        (initial (x.value heap) (y.value heap) scratch)=some final ∧
      FiniteExecution.grid plans (parameters a) (initial (x.value heap) (y.value heap) scratch)=some final.mem ∧
      (∀ m : Fin a.M,∀ n : Fin a.N,
        z.value (physical z heap (nativeValues cfg final)) (m.val*a.N+n.val)=
          nativeTarget cfg plans a (x.value heap) (y.value heap) m.val n.val) ∧
      (∀ r i,SourceGraph.GridOutside (parameters a) r i →
        SourceGraph.Native.decodeMemory cfg final.mem r i=inputMemory (x.value heap) (y.value heap) scratch.mem r i) ∧
      (∀ addr,addr<z.base ∨ z.base+z.count≤addr →
        physical z heap (nativeValues cfg final) addr=heap addr) ∧
      (∀ i,i<x.tensor.storage.elements → physical z heap (nativeValues cfg final)
        (x.tensor.storage.base/4+i)=heap (x.tensor.storage.base/4+i)) ∧
      (∀ i,i<y.tensor.storage.elements → physical z heap (nativeValues cfg final)
        (y.tensor.storage.base/4+i)=heap (y.tensor.storage.base/4+i)) := by
  obtain ⟨final,he,hfinite,hc,hf⟩ := graph_correct rest plans a h native x y z heap scratch
  refine ⟨final,he,hfinite,?_,?_,physical_frame z heap _,?_,?_⟩
  · intro m n
    rw [output_value z heap _ _ (by simpa using flat_inside a m n)]
    unfold nativeValues SourceGraph.Native.decodeMemory
    rw [hc m n]
    rfl
  · intro r i outside
    unfold SourceGraph.Native.decodeMemory
    rw [hf r i outside]
    exact congrFun (congrFun (SourceGraph.Native.decode_lift cfg
      (inputMemory (x.value heap) (y.value heap) scratch.mem)) r) i
  · exact input_storage_frame z x.tensor.storage binding.separateA heap _
  · exact input_storage_frame z y.tensor.storage binding.separateB heap _

/-- Real interpretation is separate from rounded native output. Both execute
the same complete source graph and refine the same finite integer indexing. -/
theorem real_correct (decode : ValueProofs.Word → ℝ) (rest : Algebra SourceGraph.Expr)
    (plans : SourceGraph.PlanSelector) (a : Arguments) (h : KernelContract a) (native : LaunchABI a)
    (x : Allocated [a.M,a.K]) (y : Allocated [a.M,a.N,a.K]) (z : Allocated [a.M,a.N])
    (binding : Binding a x y z) (heap : Heap) (scratch : State ValueProofs.Word) :
    ∃ final, SourceGraph.runGrid rest plans (parameters a)
        (initial (x.value heap) (y.value heap) scratch)=some final ∧
      FiniteExecution.grid plans (parameters a) (initial (x.value heap) (y.value heap) scratch)=some final.mem ∧
      (∀ m : Fin a.M,∀ n : Fin a.N,
        z.observe (physical z (fun addr => decode (heap addr)) (realValues decode final)) (m.val*a.N+n.val)=
          mathematical decode a (x.value heap) (y.value heap) m.val n.val) ∧
      (∀ r i,SourceGraph.GridOutside (parameters a) r i →
        final.mem r i=(initial (x.value heap) (y.value heap) scratch).mem r i) ∧
      (∀ addr,addr<z.base ∨ z.base+z.count≤addr →
        physical z (fun addr => decode (heap addr)) (realValues decode final) addr=decode (heap addr)) ∧
      (∀ i,i<x.tensor.storage.elements → physical z (fun addr => decode (heap addr)) (realValues decode final)
        (x.tensor.storage.base/4+i)=decode (heap (x.tensor.storage.base/4+i))) ∧
      (∀ i,i<y.tensor.storage.elements → physical z (fun addr => decode (heap addr)) (realValues decode final)
        (y.tensor.storage.base/4+i)=decode (heap (y.tensor.storage.base/4+i))) := by
  obtain ⟨final,he,hc,hf⟩ := RealProofs.grid_real_correct decode rest plans (parameters a)
    (initial (x.value heap) (y.value heap) scratch) (inputs_separate a) h.BM h.BN h.M h.N h.K
  refine ⟨final,he,?_,?_,hf,physical_frame z _ _,?_,?_⟩
  · rw [FiniteExecution.grid_source_execution rest plans _ _ (integerAgreement a native x y z),he]
    rfl
  · intro m n
    rw [output_observe z _ _ _ (by simpa using flat_inside a m n)]
    exact (hc m n).trans (mathematical_initial decode a _ _ scratch m.val n.val)
  · exact input_storage_frame z x.tensor.storage binding.separateA _ _
  · exact input_storage_frame z y.tensor.storage binding.separateB _ _

#axiomsClean graph_correct
#axiomsClean native_correct
#axiomsClean real_correct
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.KernelResult
