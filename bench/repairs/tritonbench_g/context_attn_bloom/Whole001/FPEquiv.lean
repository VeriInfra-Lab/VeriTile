import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.Host

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Public
open VeriTile VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def call (a : Host.Arguments) (r : Regions) (d : Preparation.Data) : FPSuccess.Call := ⟨a,r,d.inputs⟩

def separated (a : Host.Arguments) (r : Regions) (d : Preparation.Data) (A : FlatAlloc)
    (s : BlockState) (valid : Host.Admissible a r A s) : FPSuccess.Separate (call a r d) := by
  intro rg member
  have h := valid.separate
  simp only [call,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> tauto

/-- Actual FP32 typed source execution on staged expressions, with the exact
source casts, finite mask, dot accumulators and default FMA contraction. No
whole-dot/reduction/output or successful-source premise occurs. -/
theorem host_typed (a : Host.Arguments) (r : Regions) (d : Preparation.Data)
    (api : Preparation.LocalAPI r d) (initial : Preparation.Runtime) (owner : Nat)
    (layout : HostRun.LayoutDomain a) (A : FlatAlloc)
    (valid : Host.Admissible a r A (Preparation.prepare api initial).real)
    (base : RegionName → Nat) (abi : ABI.IndexABI a r d.inputs base)
    (jobs : List (GridIndex (Host.grid a))) (schedule : HostRun.CompleteSchedule a jobs) :
    ABI.SourceExact a r d.inputs base ∧ ∃ final,
      HostRun.native a r d api initial owner jobs=some (HostRun.finish a owner final) ∧
      FPSuccess.runGrid (call a r d) Native.scalar Native.dots jobs
        (Preparation.prepare api initial).native=some final ∧
      (∀ rg off, api.visible rg → ¬FPSuccess.writes (call a r d) jobs rg off →
        final.mem rg off=initial.native.mem rg off) ∧
      (HostRun.finish a owner final).pythonReturn=none ∧
      (HostRun.finish a owner final).outputOwner=owner ∧
      (HostRun.finish a owner final).descriptor=a.o := by
  have prepared := Preparation.prepared_inputs (a:=a) r d api initial
  obtain ⟨final,hr,metadata,hframe⟩ := FPSuccess.grid_success_frame (call a r d)
    Native.scalar Native.dots (separated a r d A _ valid) jobs _ prepared.2.1
  refine ⟨ABI.source_exact a r d.inputs base abi,final,?_,hr,?_,rfl,rfl,rfl⟩
  · change (FPSuccess.runGrid (call a r d) Native.scalar Native.dots jobs
      (Preparation.prepare api initial).native).map (HostRun.finish a owner)=_
    rw [hr]
    rfl
  · intro rg off visible outside
    exact (hframe rg off outside).trans (Preparation.caller_frame api initial rg visible off).2

def nativeWord (isa : Native.WordISA) (plan : Native.Plan) (owner : Nat)
    (s : State Native.Expr) (rg : RegionName) (off : Nat) : BitVec 32 :=
  (Native.lower plan ((s.mem rg off).read .real) owner).eval isa.interpret

/-- Every actual output owner evaluates the graph from the complete current-
memory source run. Only nine scalar/fixed-MMA ports receive numerical meaning.
The compiler's routes, operand orientation and reduction trees are structural
binding data; this theorem makes no native=Real accuracy or GPU-run claim. -/
theorem host_native (a : Host.Arguments) (r : Regions) (d : Preparation.Data)
    (api : Preparation.LocalAPI r d) (initial : Preparation.Runtime) (outputOwner : Nat)
    (layout : HostRun.LayoutDomain a) (A : FlatAlloc)
    (valid : Host.Admissible a r A (Preparation.prepare api initial).real)
    (base : RegionName → Nat) (abi : ABI.IndexABI a r d.inputs base)
    (jobs : List (GridIndex (Host.grid a))) (schedule : HostRun.CompleteSchedule a jobs)
    (isa : Native.WordISA) (plan : Native.Plan) (owner : Nat → Nat) :
    ABI.SourceExact a r d.inputs base ∧ ∃ final,
      HostRun.native a r d api initial outputOwner jobs=some (HostRun.finish a outputOwner final) ∧
      FPSuccess.runGrid (call a r d) Native.scalar Native.dots jobs
        (Preparation.prepare api initial).native=some final ∧
      (∀ idx i, FPSuccess.laneActive (call a r d) idx i →
        let off := FPSuccess.laneOffset (call a r d) idx i
        nativeWord isa plan (owner off) (HostRun.finish a outputOwner final).memory r.out off=
          (Native.lower plan ((final.mem r.out off).read .real) (owner off)).eval isa.interpret) ∧
      (∀ rg off, api.visible rg → ¬FPSuccess.writes (call a r d) jobs rg off →
        final.mem rg off=initial.native.mem rg off) ∧
      (HostRun.finish a outputOwner final).pythonReturn=none ∧
      (HostRun.finish a outputOwner final).outputOwner=outputOwner ∧
      (HostRun.finish a outputOwner final).descriptor=a.o := by
  obtain ⟨exactIndex,final,returned,executed,frame,hret,howner,hdesc⟩ :=
    host_typed a r d api initial outputOwner layout A valid base abi jobs schedule
  exact ⟨exactIndex,final,returned,executed,fun _ _ _ => rfl,frame,hret,howner,hdesc⟩

/-- Every logical batch/head/uncached-row/channel is reached by the actual
three-axis launch, including both Tesla branches, page-table tails and padded
channels. Native instruction coverage is proved from fixed MMA indexing and
ordered-tree leaf permutations, not numerical reduction or dot assumptions. -/
theorem host_native_coverage (a : Host.Arguments) (r : Regions) (d : Preparation.Data)
    (api : Preparation.LocalAPI r d) (initial : Preparation.Runtime)
    (layout : HostRun.LayoutDomain a) (A : FlatAlloc)
    (valid : Host.Admissible a r A (Preparation.prepare api initial).real)
    (base : RegionName → Nat) (abi : ABI.IndexABI a r d.inputs base)
    (jobs : List (GridIndex (Host.grid a))) (schedule : HostRun.CompleteSchedule a jobs) :
    ABI.SourceExact a r d.inputs base ∧
      (∀ (b h row channel : Nat), b<a.batch → h<a.q.heads →
        (row : Int)<d.inputs.sequence b-d.inputs.prompt b → channel<a.q.dims →
        ∃ idx∈jobs, ∃ i : RealIO.Lane (Host.layout a),
          FPSuccess.laneActive (call a r d) idx i ∧ (Host.launchOf a idx).batch=b ∧
          (Host.launchOf a idx).head=h ∧ queryRow (Host.layout a) (Host.launchOf a idx) i.1.val=row ∧
          i.2.1.val=channel ∧ FPSuccess.laneOffset (call a r d) idx i<A.extent r.out) ∧
      (∀ D e, e<D → e/8∈List.range ((D+7)/8) ∧ e%8<8 ∧ e/8*8+e%8=e) ∧
      (∀ n (tree : Native.OrderedTree n), tree.tree.leaves.Perm (List.finRange n)) := by
  have prepared := Preparation.prepared_inputs (a:=a) r d api initial
  refine ⟨ABI.source_exact a r d.inputs base abi,?_,Native.dot_feature_covered,fun _ tree => tree.complete⟩
  intro b h row channel hb hh hrow hd
  have hr : (row : Int)<(inputs (Preparation.prepare api initial).real r).sequence b-
      (inputs (Preparation.prepare api initial).real r).prompt b := by simpa only [prepared.1] using hrow
  obtain ⟨idx,i,hi,hb',hh',hr',hd'⟩ := Host.covers a r A (Preparation.prepare api initial).real valid b h row channel hb hh hr hd
  have hbnd := AddressBounds.inside_toNat ((valid.bounds idx).output i.1 i.2.1 hi.1 hi.2)
  refine ⟨idx,HostRun.covered a jobs schedule idx,i,?_,hb',hh',hr',hd',?_⟩
  · simpa only [FPSuccess.laneActive,call,Host.active,prepared.1] using hi
  · simpa only [FPSuccess.laneOffset,call,RealIO.write,prepared.1] using hbnd

#axiomsClean host_typed
#axiomsClean host_native
#axiomsClean host_native_coverage
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Public
