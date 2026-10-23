import bench.repairs.tritonbench_g.iv_dependent_matmul.Kernels
import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Ideal
import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Wrapper

/-! Complete independent Real interpretation of the repaired finite program.
The target is the logical matrix product, independent of execution registers.
All five pointer schedules, ceil-K tails, zero K, edge masks, evolving grid
memory, FP32 allocation metadata, return identity and the full frame are covered.
Real arithmetic erases numerical rounding only. No FP exactness premise occurs
in this contract; rounded native execution belongs to FPEquiv.lean. -/
namespace VeriTile.Bench.Repairs.IvDependentMatmul.Correct
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Kernels Whole
noncomputable section

specification repaired_whole_host_real (graph : Graph) (r : Request)
    (layout : State Word) (allocation : Allocation r layout)
    (addressable : Index64.Addressable r layout allocation)
    (launchable : Index64.Launchable r) (s : State ℝ) :
    (Execution.host (Execution.ideal r) graph allocation.baseA allocation.baseB allocation.baseC s).1=outputMetadata r ∧
    (∀ i<r.rows,∀ j<r.columns,
      ((Execution.host (Execution.ideal r) graph allocation.baseA allocation.baseB allocation.baseC s).2.mem
        r.C (offset r i j)).read .fp32=
      RealTarget.mathematical r.inner (Ideal.inputA r s) (Ideal.inputB r s) i j) ∧
    (∀ region o,¬Footprint r region o →
      (Execution.host (Execution.ideal r) graph allocation.baseA allocation.baseB allocation.baseC s).2.mem region o=s.mem region o) ∧
    Native.Correspondence r layout allocation := by
  obtain ⟨hm,hv,hf⟩ := Ideal.host_correct graph r layout allocation addressable launchable s
  refine ⟨hm,?_,hf,Native.correspondence r layout allocation addressable launchable⟩
  intro i hi j hj
  rw [hv i hi j hj,Cell.read_same]

/-- Whole source wrapper, including ordinary API failures. On success the
mathematical inputs are the actual random-allocation snapshots, the returned
object is the exact empty-allocation object, and caller memory is preserved. -/
specification repaired_whole_wrapper_real [Inhabited Word] {Device Error : Type}
    (graph : Graph) (r : Request) (api : Wrapper.Calls ℝ Device Error) (before : Wrapper.Heap ℝ) :
    match Wrapper.run Execution.ideal graph api r before with
    | .error e => Wrapper.FailedAt api r before e
    | .ok out => ∃ d : Device, ∃ p : Wrapper.Prepared ℝ r before, ∃ support : Index64.Launchable p.request,
      Wrapper.PreparedBy api r before d p ∧ Wrapper.launch api d p=.ok ⟨support⟩ ∧
      out=Wrapper.result Execution.ideal graph p ∧ out.tensor=p.c.tensor ∧
      out.tensor.metadata=outputMetadata p.request ∧
      (∀ i<r.rows,∀ j<r.columns,
        (out.heap.state.mem p.c.tensor.region (offset p.request i j)).read .fp32=
          RealTarget.mathematical r.inner (Ideal.inputA p.request p.a.after.state)
            (Ideal.inputB p.request p.b.after.state) i j) ∧
      (∀ region o,¬p.footprint region o → out.heap.state.mem region o=before.state.mem region o) ∧
      (∀ old∈before.buffers,∀ o,out.heap.state.mem old.region o=before.state.mem old.region o) ∧
      Native.Correspondence p.request Wrapper.layout p.realAllocation := by
  rcases Wrapper.run_cases Execution.ideal graph api r before with ⟨e,he,hf⟩ | ⟨d,p,support,he,hp,hl⟩
  · simp only [he]
    exact hf
  · simp only [he]
    obtain ⟨_,hv,hf⟩ := Ideal.host_correct graph p.request Wrapper.layout p.realAllocation
      p.realAddressable support p.c.after.state
    obtain ⟨hpre,hcaller⟩ := Wrapper.caller_frame p _ hf
    have ha : Ideal.inputA p.request p.c.after.state=Ideal.inputA p.request p.a.after.state := by
      funext i k
      exact congrArg (fun cell => cell.read .fp32) (p.memoryA (i*max r.inner 1+k))
    have hb : Ideal.inputB p.request p.c.after.state=Ideal.inputB p.request p.b.after.state := by
      funext k j
      exact congrArg (fun cell => cell.read .fp32) (p.memoryB (k*max r.columns 1+j))
    refine ⟨d,p,support,hp,hl,rfl,rfl,rfl,?_,hpre,hcaller,
      Native.correspondence p.request Wrapper.layout p.realAllocation p.realAddressable support⟩
    intro i hi j hj
    have h := congrArg (fun cell => cell.read .fp32) (hv i hi j hj)
    simpa only [Cell.read_same,ha,hb] using h

#axiomsClean repaired_whole_wrapper_real
#axiomsClean repaired_whole_host_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IvDependentMatmul.Correct
