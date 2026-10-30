import bench.tritonbench_g.iv_dependent_matmul.Proofs.CheckedHost
import bench.tritonbench_g.iv_dependent_matmul.Proofs.ByteExecution

namespace VeriTile.Bench.Repairs.IvDependentMatmul.ByteHost
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open OutputBytes
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 1000000

def aView {r : Request} {before : Wrapper.Heap Word} (p : Wrapper.Prepared Word r before) : View :=
  ⟨p.a.tensor.region,p.a.tensor.base,r.rows*r.inner⟩
def bView {r : Request} {before : Wrapper.Heap Word} (p : Wrapper.Prepared Word r before) : View :=
  ⟨p.b.tensor.region,p.b.tensor.base,r.inner*r.columns⟩
def cView {r : Request} {before : Wrapper.Heap Word} (p : Wrapper.Prepared Word r before) : View :=
  ByteExecution.view p.request p.c.tensor.base
def Outside (v : View) (address : Nat) : Prop := address<v.base ∨ v.base+4*v.extent≤address

/-- Ordinary physical allocation/readback law for the actual A/B/C objects.
It contains no kernel result, source envelope, or launch-success condition. -/
structure Allocation {r : Request} {before : Wrapper.Heap Word}
    (p : Wrapper.Prepared Word r before) (original prepared : Read) : Prop where
  inputA : Bound prepared (aView p) p.c.after.state
  inputB : Bound prepared (bView p) p.c.after.state
  outputC : Bound prepared (cView p) p.c.after.state
  frame : ∀ address,Outside (aView p) address → Outside (bView p) address → Outside (cView p) address →
    prepared address=original address

def result (arithmetic : (r : Request) → Execution.Arithmetic Word r) (g : Graph)
    {r : Request} {before : Wrapper.Heap Word} (p : Wrapper.Prepared Word r before) (read : Read) :
    Wrapper.Returned Word r × Read :=
  let final := ByteExecution.grid (arithmetic p.request) g p.a.tensor.base p.b.tensor.base p.c.tensor.base
    (List.range (programs p.request)) (p.c.after.state,read)
  ({CheckedHost.result arithmetic g p with heap:={p.c.after with state:=final.1}},final.2)

def call {Device Error : Type} (arithmetic : (r : Request) → Execution.Arithmetic Word r) (g : Graph)
    (api : Wrapper.Calls Word Device Error) (r : Request) (before : Wrapper.Heap Word)
    (bytesFor : Wrapper.Prepared Word r before → Read) (defaultFP32 cuda : Bool) :
    Option (Wrapper.Returned Word r × Read) :=
  if Contract.check r defaultFP32 cuda=.ok () then
    (Wrapper.prepare api r before).toOption.bind fun (d,p) =>
      if CheckedHost.Empty r then some (result arithmetic g p (bytesFor p))
      else (Wrapper.launch api d p).toOption.map fun _ => result arithmetic g p (bytesFor p)
  else none

theorem call_prepared {Device Error : Type} (arithmetic : (r : Request) → Execution.Arithmetic Word r) (g : Graph)
    (api : Wrapper.Calls Word Device Error) (r : Request) (before : Wrapper.Heap Word)
    (bytesFor : Wrapper.Prepared Word r before → Read) (defaultFP32 cuda : Bool)
    (h : Contract.Preconditions r defaultFP32 cuda) (d : Device) (p : Wrapper.Prepared Word r before)
    (hp : Wrapper.PreparedBy api r before d p) (hl : CheckedHost.LaunchReady api d p) :
    call arithmetic g api r before bytesFor defaultFP32 cuda=some (result arithmetic g p (bytesFor p)) := by
  have prepared : Wrapper.prepare api r before=.ok (d,p) := by
    simp only [Wrapper.prepare,hp.1,hp.2.1,hp.2.2.1,hp.2.2.2]
  have checked := (Contract.check_iff r defaultFP32 cuda).mpr h
  by_cases empty : CheckedHost.Empty r
  · simp only [call,if_pos checked,prepared,Except.toOption,Option.bind_some,if_pos empty]
  · obtain ⟨support,launch⟩ := hl empty
    simp only [call,if_pos checked,prepared,Except.toOption,Option.bind_some,if_neg empty,launch,Option.map_some]

theorem result_render (arithmetic : (r : Request) → Execution.Arithmetic Word r) (g : Graph)
    {r : Request} {before : Wrapper.Heap Word} (p : Wrapper.Prepared Word r before)
    (h : Contract.Admitted r) (read : Read) (bound : Bound read (cView p) p.c.after.state) :
    result arithmetic g p read=
      (CheckedHost.result arithmetic g p,
        render read (cView p) (CheckedHost.result arithmetic g p).heap.state) := by
  let allocation := p.allocation p.c.after.state p.typedA p.typedB
  have d := p.addressable p.c.after.state p.typedA p.typedB
  have l := Contract.launchable p.request (CheckedHost.prepared_admitted p h)
  have grid := ByteExecution.grid_render (arithmetic p.request) g p.c.after.state allocation d l read
    (List.range (programs p.request)) (fun _ hi => List.mem_range.mp hi) p.c.after.state
  change ByteExecution.grid (arithmetic p.request) g p.a.tensor.base p.b.tensor.base p.c.tensor.base
      (List.range (programs p.request)) (p.c.after.state,render read (cView p) p.c.after.state)=_ at grid
  rw [render_initial read (cView p) p.c.after.state bound] at grid
  simp only [result,grid,CheckedHost.result,Wrapper.result,Execution.host]
  rfl

structure Post {r : Request} {before : Wrapper.Heap Word} (p : Wrapper.Prepared Word r before)
    (original prepared final : Read) (value : Nat → Nat → Word) : Prop where
  aSnapshot : ∀ i,i<r.rows*r.inner → read32 prepared (p.a.tensor.base+4*i)=
    ((p.a.after.state.mem p.a.tensor.region i).read .fp32).bits
  bSnapshot : ∀ i,i<r.inner*r.columns → read32 prepared (p.b.tensor.base+4*i)=
    ((p.b.after.state.mem p.b.tensor.region i).read .fp32).bits
  words : ∀ i,i<r.rows → ∀ j,j<r.columns →
    read32 final (p.c.tensor.base+4*offset p.request i j)=(value i j).bits
  bytes : ∀ i,i<r.rows → ∀ j,j<r.columns → ∀ lane,lane<4 →
    final (p.c.tensor.base+4*offset p.request i j+lane)=byte (value i j).bits lane
  addresses : ∀ i,i<r.rows → ∀ j,j<r.columns → ∀ lane,lane<4 →
    p.c.tensor.base+4*offset p.request i j+lane<2^64
  inputA : ∀ address,p.a.tensor.base≤address → address<p.a.tensor.base+4*(r.rows*r.inner) → final address=prepared address
  inputB : ∀ address,p.b.tensor.base≤address → address<p.b.tensor.base+4*(r.inner*r.columns) → final address=prepared address
  frame : ∀ address,Outside (aView p) address → Outside (bView p) address → Outside (cView p) address → final address=original address

theorem rendered_post {r : Request} {before : Wrapper.Heap Word} (p : Wrapper.Prepared Word r before)
    (original prepared : Read) (allocation : Allocation p original prepared)
    (out : Wrapper.Returned Word r) (value : Nat → Nat → Word) (logical : CheckedHost.Post p out value) :
    Post p original prepared (render prepared (cView p) out.heap.state) value := by
  have size : (cView p).extent=r.rows*r.columns := rfl
  have base : (cView p).base=p.c.tensor.base := rfl
  have region : (cView p).region=p.c.tensor.region := rfl
  refine ⟨?_,?_,?_,?_,?_,?_,?_,?_⟩
  · intro i hi
    have words := allocation.inputA i hi
    change p.c.after.state.mem p.a.tensor.region i=Cell.mk .fp32 ⟨read32 prepared (p.a.tensor.base+4*i)⟩ at words
    rw [p.memoryA] at words
    rw [words]
    rfl
  · intro i hi
    have words := allocation.inputB i hi
    change p.c.after.state.mem p.b.tensor.region i=Cell.mk .fp32 ⟨read32 prepared (p.b.tensor.base+4*i)⟩ at words
    rw [p.memoryB] at words
    rw [words]
    rfl
  · intro i hi j hj
    have bound : offset p.request i j<(cView p).extent := output_bound p.request i j hi hj
    rw [←base,render_read prepared (cView p) out.heap.state _ bound,region,logical.values i hi j hj]
    rfl
  · intro i hi j hj lane hl
    have bound : offset p.request i j<(cView p).extent := output_bound p.request i j hi hj
    rw [←base,render_at prepared (cView p) out.heap.state _ lane bound hl,region,logical.values i hi j hj]
    rfl
  · intro i hi j hj lane hl
    have bound := output_bound p.request i j hi hj
    have addressable := p.c.addressable
    simp only [List.prod_cons,List.prod_nil,Nat.mul_one] at addressable
    change offset p.request i j<r.rows*r.columns at bound
    omega
  · intro address low high
    apply render_frame
    have separate := p.c.physical p.a.tensor.buffer (by rw [p.b.registry,p.a.registry]; simp)
    simp only [Wrapper.Tensor.buffer,List.prod_cons,List.prod_nil,Nat.mul_one] at separate
    change address<p.c.tensor.base ∨ p.c.tensor.base+4*(r.rows*r.columns)≤address
    omega
  · intro address low high
    apply render_frame
    have separate := p.c.physical p.b.tensor.buffer (by rw [p.b.registry]; simp)
    simp only [Wrapper.Tensor.buffer,List.prod_cons,List.prod_nil,Nat.mul_one] at separate
    change address<p.c.tensor.base ∨ p.c.tensor.base+4*(r.rows*r.columns)≤address
    omega
  · intro address ha hb hc
    exact (render_frame prepared (cView p) out.heap.state address hc).trans (allocation.frame address ha hb hc)

#axiomsClean call_prepared
#axiomsClean result_render
#axiomsClean rendered_post
end
end VeriTile.Bench.Repairs.IvDependentMatmul.ByteHost
