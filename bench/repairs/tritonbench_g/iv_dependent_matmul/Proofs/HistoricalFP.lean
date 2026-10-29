import bench.repairs.tritonbench_g.iv_dependent_matmul.Kernels
import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Conformance
import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Wrapper

/-! The finite repaired execution preserves the ordered FP reference at all
returned FP32 cells and preserves the complete memory frame. Operands and
accumulation are FP32; explicit FP16 narrowing and FP32 widening remain.
Dot and DotAcc callbacks are parameters of the corresponding primitive contract.
Source Dot+Add is never identified with compiled DotAcc. The software instance
is explicitly sequential and is not a native TF32 tensor-core conformance claim.
All five pointer schedules share this contract; there is no exact-Real premise. -/
namespace VeriTile.Bench.Repairs.IvDependentMatmul.FPEquiv
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Kernels Whole
noncomputable section

specification repaired_whole_host_values [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (D : Dot) (F : DotAcc) (graph : Graph) (r : Request) (s : State Word)
    (allocation : Allocation r s) (addressable : Index64.Addressable r s allocation)
    (launchable : Index64.Launchable r) :
    (Execution.host (Execution.fp cfg D F r) graph allocation.baseA allocation.baseB allocation.baseC s).1=outputMetadata r ∧
    Result cfg D F graph r s
      (Execution.host (Execution.fp cfg D F r) graph allocation.baseA allocation.baseB allocation.baseC s).2 ∧
    (∃ final,Whole.host rest cfg D F graph r s=some (outputMetadata r,final) ∧
      final.mem=(Execution.host (Execution.fp cfg D F r) graph allocation.baseA allocation.baseB allocation.baseC s).2.mem) ∧
    Native.Correspondence r s allocation := by
  obtain ⟨final,he,hm,hv⟩ := Execution.host_projection rest cfg D F graph r s allocation addressable launchable
  exact ⟨rfl,hv,⟨final,he,hm⟩,Native.correspondence r s allocation addressable launchable⟩

specification repaired_sequential_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (graph : Graph) (r : Request) (s : State Word) (allocation : Allocation r s)
    (addressable : Index64.Addressable r s allocation) (launchable : Index64.Launchable r) :
    Result cfg (softwareDot cfg) (softwareDotAcc cfg) graph r s
      (Execution.host (Execution.fp cfg (softwareDot cfg) (softwareDotAcc cfg) r) graph
        allocation.baseA allocation.baseB allocation.baseC s).2 :=
  (repaired_whole_host_values rest cfg (softwareDot cfg) (softwareDotAcc cfg) graph r s allocation addressable launchable).2.1

specification repaired_schedule_equivalence [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (D : Dot) (F : DotAcc) (graph : Graph) (r : Request) (s : State Word)
    (allocation : Allocation r s) (addressable : Index64.Addressable r s allocation)
    (launchable : Index64.Launchable r) (a b : Schedule) :
    (Execution.host (Execution.fp cfg D F {r with schedule:=a}) graph
      allocation.baseA allocation.baseB allocation.baseC s).2.mem=
    (Execution.host (Execution.fp cfg D F {r with schedule:=b}) graph
      allocation.baseA allocation.baseB allocation.baseC s).2.mem := by
  let aa : Allocation {r with schedule:=a} s :=
    ⟨allocation.freshA,allocation.freshB,allocation.baseA,allocation.baseB,allocation.baseC,
      allocation.physicalFreshA,allocation.physicalFreshB,allocation.alignA,allocation.alignB,allocation.alignC,
      allocation.inputA,allocation.inputB⟩
  let ab : Allocation {r with schedule:=b} s :=
    ⟨allocation.freshA,allocation.freshB,allocation.baseA,allocation.baseB,allocation.baseC,
      allocation.physicalFreshA,allocation.physicalFreshB,allocation.alignA,allocation.alignB,allocation.alignC,
      allocation.inputA,allocation.inputB⟩
  have da : Index64.Addressable {r with schedule:=a} s aa :=
    ⟨addressable.input,addressable.weight,addressable.output⟩
  have db : Index64.Addressable {r with schedule:=b} s ab :=
    ⟨addressable.input,addressable.weight,addressable.output⟩
  have la : Index64.Launchable {r with schedule:=a} :=
    ⟨launchable.rowsBlock,launchable.columnsBlock,launchable.innerBlock,launchable.grid⟩
  have lb : Index64.Launchable {r with schedule:=b} :=
    ⟨launchable.rowsBlock,launchable.columnsBlock,launchable.innerBlock,launchable.grid⟩
  have ha := (repaired_whole_host_values rest cfg D F graph {r with schedule:=a} s aa da la).2.1
  have hb := (repaired_whole_host_values rest cfg D F graph {r with schedule:=b} s ab db lb).2.1
  exact result_memory_unique cfg D F graph r s _ _ ha hb

/-- Numerical port: only primitive bit equalities on reached operands are
assumed. The reference callbacks must be selected for the actual production
precision and MMA configuration; no native discharge is claimed here. -/
specification repaired_native_under_local_calls [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (D : Dot) (F : DotAcc) (graph : Graph) (r : Request) (s : State Word)
    (allocation : Allocation r s) (addressable : Index64.Addressable r s allocation)
    (launchable : Index64.Launchable r) (actual : Execution.Arithmetic Word r)
    (numerical : Conformance.LaunchCalls actual (Execution.fp cfg D F r) graph
      allocation.baseA allocation.baseB allocation.baseC (List.range (programs r)) s) :
    Result cfg D F graph r s
      (Execution.host actual graph allocation.baseA allocation.baseB allocation.baseC s).2 := by
  rw [Conformance.host_equal actual (Execution.fp cfg D F r) graph
    allocation.baseA allocation.baseB allocation.baseC s numerical]
  exact (repaired_whole_host_values rest cfg D F graph r s allocation addressable launchable).2.1

/-- Complete FP wrapper outcome from caller memory. The ordered reference
reads the actual A/B state after random allocation and before empty C; the
wrapper itself invokes the finite host and returns the exact allocated C. -/
specification repaired_whole_wrapper_values [Inhabited Word] {Device Error : Type}
    (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc) (graph : Graph)
    (r : Request) (api : Wrapper.Calls Word Device Error) (before : Wrapper.Heap Word) :
    match Wrapper.run (Execution.fp cfg D F) graph api r before with
    | .error e => Wrapper.FailedAt api r before e
    | .ok out => ∃ d : Device, ∃ p : Wrapper.Prepared Word r before, ∃ support : Index64.Launchable p.request,
      Wrapper.PreparedBy api r before d p ∧ Wrapper.launch api d p=.ok ⟨support⟩ ∧
      out=Wrapper.result (Execution.fp cfg D F) graph p ∧ out.tensor=p.c.tensor ∧
      out.tensor.metadata=outputMetadata p.request ∧
      (∀ i<r.rows,∀ j<r.columns,
        out.heap.state.mem p.c.tensor.region (offset p.request i j)=
          Cell.mk .fp32 (reference cfg D F graph p.request p.b.after.state i j)) ∧
      (∀ region o,¬p.footprint region o → out.heap.state.mem region o=before.state.mem region o) ∧
      (∀ old∈before.buffers,∀ o,out.heap.state.mem old.region o=before.state.mem old.region o) ∧
      Native.Correspondence p.request p.c.after.state (p.allocation p.c.after.state p.typedA p.typedB) := by
  rcases Wrapper.run_cases (Execution.fp cfg D F) graph api r before with ⟨e,he,hf⟩ | ⟨d,p,support,he,hp,hl⟩
  · simp only [he]
    exact hf
  · simp only [he]
    let allocation := p.allocation p.c.after.state p.typedA p.typedB
    have addressable := p.addressable p.c.after.state p.typedA p.typedB
    have hv := (repaired_whole_host_values rest cfg D F graph p.request p.c.after.state
      allocation addressable support).2.1
    obtain ⟨hpre,hcaller⟩ := Wrapper.caller_frame p _ hv.2
    refine ⟨d,p,support,hp,hl,rfl,rfl,rfl,?_,hpre,hcaller,
      Native.correspondence p.request p.c.after.state allocation addressable support⟩
    intro i hi j hj
    have href : reference cfg D F graph p.request p.c.after.state i j=
        reference cfg D F graph p.request p.b.after.state i j := by
      unfold reference
      rw [accumulated_inputs cfg D F graph p.request p.b.after.state p.c.after.state _
        (fun o => p.c.frame _ o (by intro h; exact p.ca h.1.symm)) (p.memoryB)]
    have h := hv.1 i hi j hj
    change _=Cell.mk .fp32 (reference cfg D F graph p.request p.c.after.state i j) at h
    rw [href] at h
    exact h

/-- Native conditional wrapper result. Only the local primitive bit fields at
reached wrapper states are deferred; successful execution, returned identity,
allocation composition, output readback and pre-call framing are proved. -/
specification repaired_whole_wrapper_native_under_local_calls [Inhabited Word] {Device Error : Type}
    (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc) (graph : Graph)
    (r : Request) (api : Wrapper.Calls Word Device Error) (before : Wrapper.Heap Word)
    (actual : (r : Request) → Execution.Arithmetic Word r)
    (numerical : Conformance.WrapperCalls actual (Execution.fp cfg D F) graph api r before) :
    match Wrapper.run actual graph api r before with
    | .error e => Wrapper.FailedAt api r before e
    | .ok out => ∃ d : Device, ∃ p : Wrapper.Prepared Word r before, ∃ support : Index64.Launchable p.request,
      Wrapper.PreparedBy api r before d p ∧ Wrapper.launch api d p=.ok ⟨support⟩ ∧
      out=Wrapper.result actual graph p ∧ out.tensor=p.c.tensor ∧
      out.tensor.metadata=outputMetadata p.request ∧
      (∀ i<r.rows,∀ j<r.columns,
        out.heap.state.mem p.c.tensor.region (offset p.request i j)=
          Cell.mk .fp32 (reference cfg D F graph p.request p.b.after.state i j)) ∧
      (∀ region o,¬p.footprint region o → out.heap.state.mem region o=before.state.mem region o) ∧
      (∀ old∈before.buffers,∀ o,out.heap.state.mem old.region o=before.state.mem old.region o) ∧
      Native.Correspondence p.request p.c.after.state (p.allocation p.c.after.state p.typedA p.typedB) := by
  rcases Wrapper.run_cases actual graph api r before with ⟨e,he,hf⟩ | ⟨d,p,support,he,hp,hl⟩
  · simp only [he]
    exact hf
  · simp only [he]
    let allocation := p.allocation p.c.after.state p.typedA p.typedB
    have addressable := p.addressable p.c.after.state p.typedA p.typedB
    have hv := (repaired_whole_host_values rest cfg D F graph p.request p.c.after.state
      allocation addressable support).2.1
    obtain ⟨hpre,hcaller⟩ := Wrapper.caller_frame p _ hv.2
    have hh := Conformance.host_equal (actual p.request) (Execution.fp cfg D F p.request) graph
      p.a.tensor.base p.b.tensor.base p.c.tensor.base p.c.after.state
      (numerical d p hp support hl)
    refine ⟨d,p,support,hp,hl,rfl,rfl,rfl,?_,?_,?_,
      Native.correspondence p.request p.c.after.state allocation addressable support⟩
    · intro i hi j hj
      have href : reference cfg D F graph p.request p.c.after.state i j=
          reference cfg D F graph p.request p.b.after.state i j := by
        unfold reference
        rw [accumulated_inputs cfg D F graph p.request p.b.after.state p.c.after.state _
          (fun o => p.c.frame _ o (by intro h; exact p.ca h.1.symm)) (p.memoryB)]
      have h := hv.1 i hi j hj
      change _=Cell.mk .fp32 (reference cfg D F graph p.request p.c.after.state i j) at h
      rw [href] at h
      simpa only [Wrapper.result,hh] using h
    · simpa only [Wrapper.result,hh] using hpre
    · simpa only [Wrapper.result,hh] using hcaller

#axiomsClean repaired_whole_wrapper_native_under_local_calls
#axiomsClean repaired_whole_wrapper_values
#axiomsClean repaired_schedule_equivalence
#axiomsClean repaired_native_under_local_calls
#axiomsClean repaired_whole_host_values
#axiomsClean repaired_sequential_software
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IvDependentMatmul.FPEquiv
