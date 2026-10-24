import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Contract
import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Ideal
import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Wrapper

namespace VeriTile.Bench.Repairs.IvDependentMatmul.CheckedHost
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 6000000

def Empty (r : Request) : Prop := r.rows=0 ∨ r.columns=0
instance (r : Request) : Decidable (Empty r) := inferInstanceAs (Decidable (_ ∨ _))

def result {α : Type} [Inhabited α] (arithmetic : (r : Request) → Execution.Arithmetic α r)
    (g : Graph) {r : Request} {before : Wrapper.Heap α} (p : Wrapper.Prepared α r before) : Wrapper.Returned α r :=
  {Wrapper.result arithmetic g p with trace :=
    (if (CheckedHost.Empty r) then [.device,.randomA,.randomB,.emptyC,.returned]
    else [.device,.randomA,.randomB,.emptyC,.launch,.returned])}

def call {α Device Error : Type} [Inhabited α]
    (arithmetic : (r : Request) → Execution.Arithmetic α r) (g : Graph)
    (api : Wrapper.Calls α Device Error) (r : Request) (before : Wrapper.Heap α)
    (defaultFP32 cuda : Bool) : Option (Wrapper.Returned α r) :=
  if Contract.check r defaultFP32 cuda=.ok () then
    (Wrapper.prepare api r before).toOption.bind fun (d,p) =>
      if Empty r then some (result arithmetic g p)
      else (Wrapper.launch api d p).toOption.map fun _ => result arithmetic g p
  else none

def LaunchReady {α Device Error : Type} (api : Wrapper.Calls α Device Error)
    {r : Request} {before : Wrapper.Heap α} (d : Device) (p : Wrapper.Prepared α r before) : Prop :=
  ¬Empty r → ∃ support,Wrapper.launch api d p=.ok support

theorem call_prepared {α Device Error : Type} [Inhabited α]
    (arithmetic : (r : Request) → Execution.Arithmetic α r) (g : Graph)
    (api : Wrapper.Calls α Device Error) (r : Request) (before : Wrapper.Heap α)
    (defaultFP32 cuda : Bool) (h : Contract.Preconditions r defaultFP32 cuda)
    (d : Device) (p : Wrapper.Prepared α r before) (hp : Wrapper.PreparedBy api r before d p)
    (hl : LaunchReady api d p) :
    call arithmetic g api r before defaultFP32 cuda=some (result arithmetic g p) := by
  have prepared : Wrapper.prepare api r before=.ok (d,p) := by
    simp only [Wrapper.prepare,hp.1,hp.2.1,hp.2.2.1,hp.2.2.2]
  have checked := (Contract.check_iff r defaultFP32 cuda).mpr h
  by_cases empty : Empty r
  · simp only [call,if_pos checked,prepared,Except.toOption,Option.bind_some,if_pos empty]
  · obtain ⟨support,launch⟩ := hl empty
    simp only [call,if_pos checked,prepared,Except.toOption,Option.bind_some,if_neg empty,launch,Option.map_some]

theorem prepared_admitted {α : Type} {r : Request} {before : Wrapper.Heap α}
    (p : Wrapper.Prepared α r before) (h : Contract.Admitted r) : Contract.Admitted p.request :=
  Contract.prepared r p.request h ⟨rfl,rfl,rfl⟩ ⟨rfl,rfl,rfl⟩

structure Post {α : Type} {r : Request} {before : Wrapper.Heap α}
    (p : Wrapper.Prepared α r before) (out : Wrapper.Returned α r) (value : Nat → Nat → α) : Prop where
  identity : out.tensor=p.c.tensor
  metadata : out.tensor.metadata=outputMetadata p.request
  values : ∀ i,i<r.rows → ∀ j,j<r.columns →
    out.heap.state.mem p.c.tensor.region (offset p.request i j)=Cell.mk .fp32 (value i j)
  frame : ∀ region o,¬p.footprint region o → out.heap.state.mem region o=before.state.mem region o
  caller : ∀ old∈before.buffers,∀ o,out.heap.state.mem old.region o=before.state.mem old.region o

theorem real_post {r : Request} {before : Wrapper.Heap ℝ} (p : Wrapper.Prepared ℝ r before)
    (h : Contract.Admitted r) (g : Graph) :
    Post p (result Execution.ideal g p)
      (RealTarget.mathematical r.inner (Ideal.inputA p.request p.a.after.state)
        (Ideal.inputB p.request p.b.after.state)) := by
  have launch := Contract.launchable p.request (prepared_admitted p h)
  obtain ⟨_,hv,hf⟩ := Ideal.host_correct g p.request Wrapper.layout p.realAllocation p.realAddressable launch p.c.after.state
  obtain ⟨frame,caller⟩ := Wrapper.caller_frame p _ hf
  have ha : Ideal.inputA p.request p.c.after.state=Ideal.inputA p.request p.a.after.state := by
    funext i k
    exact congrArg (fun cell => cell.read .fp32) (p.memoryA (i*max r.inner 1+k))
  have hb : Ideal.inputB p.request p.c.after.state=Ideal.inputB p.request p.b.after.state := by
    funext k j
    exact congrArg (fun cell => cell.read .fp32) (p.memoryB (k*max r.columns 1+j))
  refine ⟨rfl,rfl,?_,frame,caller⟩
  intro i hi j hj
  simpa only [Ideal.Good,ha,hb] using hv i hi j hj

theorem fp_post (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    {r : Request} {before : Wrapper.Heap Word} (p : Wrapper.Prepared Word r before)
    (h : Contract.Admitted r) (g : Graph) :
    Post p (result (Execution.fp cfg D F) g p)
      (reference cfg D F g p.request p.b.after.state) := by
  let allocation := p.allocation p.c.after.state p.typedA p.typedB
  have addressable := p.addressable p.c.after.state p.typedA p.typedB
  have launch := Contract.launchable p.request (prepared_admitted p h)
  obtain ⟨final,he,hm,hv⟩ := Execution.host_projection rest cfg D F g p.request p.c.after.state allocation addressable launch
  obtain ⟨frame,caller⟩ := Wrapper.caller_frame p _ hv.2
  refine ⟨rfl,rfl,?_,frame,caller⟩
  intro i hi j hj
  have href : reference cfg D F g p.request p.c.after.state i j=reference cfg D F g p.request p.b.after.state i j := by
    unfold reference
    rw [accumulated_inputs cfg D F g p.request p.b.after.state p.c.after.state _
      (fun o => p.c.frame _ o (by intro he; exact p.ca he.1.symm)) p.memoryB]
  simpa only [Good,href] using hv.1 i hi j hj

theorem empty_programs (r : Request) (empty : Empty r) : programs r=0 := by
  have bm := r.positiveRowsBlock
  have bn := r.positiveColumnsBlock
  rcases empty with e | e
  · simp only [programs,tileRows,e,Nat.zero_add]
    rw [Nat.div_eq_of_lt (by omega : r.blockRows-1<r.blockRows)]
    simp
  · simp only [programs,tileColumns,e,Nat.zero_add]
    rw [Nat.div_eq_of_lt (by omega : r.blockColumns-1<r.blockColumns)]
    simp

#axiomsClean call_prepared
#axiomsClean real_post
#axiomsClean fp_post
#axiomsClean empty_programs
end
end VeriTile.Bench.Repairs.IvDependentMatmul.CheckedHost
