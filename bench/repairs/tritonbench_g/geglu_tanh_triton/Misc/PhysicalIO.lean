import bench.repairs.tritonbench_g.geglu_tanh_triton.Misc.NativeIO
import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.NativeSource

/-! Concrete bindings from every original physical Ops heap. No allocator,
canonical byte heap, extra alias exclusion or memory-representation restriction
is introduced. Existing physical source certificates and byte frames remain
the transport foundation in RawKernel. -/
namespace VeriTile.Bench.Repairs.GegluTanhTriton.PhysicalIO
open VeriTile Triton FP.Structural Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def values {M α β : Type} (ops : Ops M α β) (p : Layout) (entry : M) :
    ChannelValues α [⟨.real,[p.count],"A"⟩,⟨.real,[p.count],"B"⟩,⟨.real,[p.count],"DC"⟩] :=
  fun i=>Fin.cases (fun j=>ops.read entry (p.key .a j.1.val))
    (Fin.cases (fun j=>ops.read entry (p.key .b j.1.val))
      (Fin.cases (fun j=>ops.read entry (p.key .third j.1.val)) (fun i=>Fin.elim0 i))) i

def bounds (backward : Bool) (p : Layout) (r : RegionName) : Nat :=
  if r="A" then (p.allocation .a).words
  else if r="B" then (p.allocation .b).words
  else if r=Source.region backward .third then (p.allocation .third).words
  else 0

theorem real_binding {M α β : Type} [Inhabited α] (ops : Ops M α β) (algebra : Algebra α)
    (candidate backward : Bool) (p : Layout) (order : List Nat) (entry : M) :
    (IO.program algebra candidate backward p.count order id id).InputBinding
      (values ops p entry) (Source.state ops backward p entry) := by
  intro i j active
  fin_cases i <;> cases backward
  all_goals simp [IO.program,values,Source.state,Source.region,Cell.read,mapValue] at *
  all_goals rfl

theorem native_binding {M β : Type} (ops : Ops M Native.Word β) (instructions : Native.Instructions)
    (candidate backward : Bool) (p : Layout) (order : List Nat) (entry : M) :
    (Reference.nativeIO instructions candidate backward p.count order).InputBinding
      (values ops p entry) (Native.graphState ops backward p entry) := by
  intro i j active
  fin_cases i <;> cases backward
  all_goals simp [Reference.nativeIO,IO.program,values,Native.graphState,Source.region,Cell.read,mapValue] at *
  all_goals rfl

theorem finite_bounds {α β : Type} (algebra : Algebra β) (candidate backward : Bool)
    (p : Layout) (valid : Valid p) (order : List Nat) (encode : α → β) (observe : β → α)
    (vals : (IO.program algebra candidate backward p.count order encode observe).Inputs) :
    (IO.program algebra candidate backward p.count order encode observe).InBounds (bounds backward p) vals := by
  refine ⟨?_,?_,?_,?_⟩
  · intro r outside
    cases backward <;> simp [IO.program] at outside
    all_goals simp [bounds,Source.region,outside.1,outside.2.1,outside.2.2.1,outside.2.2.2]
  · intro i j active
    fin_cases i
    all_goals have a := lt_of_lt_of_le j.1.isLt (valid.capacity .a)
    all_goals have b := lt_of_lt_of_le j.1.isLt (valid.capacity .b)
    all_goals have c := lt_of_lt_of_le j.1.isLt (valid.capacity .third)
    all_goals cases backward
    all_goals simp [IO.program,bounds,Source.region] at *
    all_goals assumption
  · intro o j active
    fin_cases o
    all_goals have a := lt_of_lt_of_le j.1.isLt (valid.capacity .a)
    all_goals have b := lt_of_lt_of_le j.1.isLt (valid.capacity .b)
    all_goals have c := lt_of_lt_of_le j.1.isLt (valid.capacity .third)
    all_goals cases backward
    all_goals simp [IO.program,IO.channel,enabled,bounds,Source.region] at *
    all_goals assumption
  · intro i
    exact Fin.elim0 i

#axiomsClean real_binding
#axiomsClean native_binding
#axiomsClean finite_bounds
end
end VeriTile.Bench.Repairs.GegluTanhTriton.PhysicalIO
