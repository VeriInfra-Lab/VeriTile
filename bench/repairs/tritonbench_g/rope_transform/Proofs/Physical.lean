import bench.repairs.tritonbench_g.rope_transform.Proofs.NativeHost
import bench.repairs.tritonbench_g.rope_transform.Proofs.Guards
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.RopeTransform.Physical
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 4000000

abbrev Byte := BitVec 8
abbrev Job := Bool × Nat

def key (t : Host.Tensor) (o : Nat) : Key := ⟨t.device,t.base+4*o⟩
def jobs (r : Request) : List Job :=
  (List.range (r.rows*r.stride Bool.false)).map (Bool.false,·) ++
  (List.range (r.rows*r.stride Bool.true)).map (Bool.true,·)

theorem member (r : Request) (j : Job) : j∈jobs r ↔ j.2<r.rows*r.stride j.1 := by
  rcases j with ⟨role,o⟩
  cases role <;> simp [jobs]

theorem keys_apart (r : Request) (call : Host.Preparation Native.Term r)
    (a b : Job) (ha : a∈jobs r) (hb : b∈jobs r) (different : a≠b) :
    (key (call.output a.1) a.2).disjoint (key (call.output b.1) b.2) := by
  have hao := (member r a).mp ha
  have hbo := (member r b).mp hb
  by_cases same : a.1=b.1
  · have dne : a.2≠b.2 := by intro h; exact different (Prod.ext same h)
    simp only [Key.disjoint,key,same]
    by_cases lt : a.2<b.2
    · exact Or.inr (Or.inl (by omega))
    · exact Or.inr (Or.inr (by omega))
  · obtain before | after := call.outputApart a.1 b.1 same
    · exact Or.inr (Or.inl (by simp only [key]; omega))
    · exact Or.inr (Or.inr (by simp only [key]; omega))

/-- Byte interpretation of the successful typed source's FP32 stores. Every
word is obtained from the executed source state. The map contains no RoPE
formula or result callback. Distinct source stores own disjoint four-byte
spans, so materializing their final words preserves the source store result. -/
def materialize {M : Type} (ops : Ops M Native.Word Byte) (I : Native.Instructions)
    (S : Native.Schedule) (r : Request) (call : Host.Preparation Native.Term r)
    (source : State Native.Term) (initial : M) : M :=
  stores ops (fun j : Job => key (call.output j.1) j.2)
    (fun j => Native.observe I S source (region j.1) j.2) (jobs r) initial

theorem read_materialize {M : Type} (ops : Ops M Native.Word Byte) (I : Native.Instructions)
    (S : Native.Schedule) (r : Request) (call : Host.Preparation Native.Term r)
    (source : State Native.Term) (initial : M) (role : Bool) (o : Nat)
    (active : o<r.rows*r.stride role) :
    ops.read (materialize ops I S r call source initial) (key (call.output role) o)=
      Native.observe I S source (region role) o := by
  unfold materialize
  apply VeriTile.Bench.Repairs.SwigluTriton.Memory.readback ops _ _ _ _ (role,o) ((member r (role,o)).mpr active)
  · intro j hj equal
    have eq : j=(role,o) := by
      by_contra ne
      exact Key.disjoint_ne (keys_apart r call j (role,o) hj
        ((member r (role,o)).mpr active) ne) equal
    rw [eq]
  · intro j hj different
    exact keys_apart r call (role,o) j ((member r (role,o)).mpr active) hj
      (by intro same; subst j; exact different rfl)

def Outside (r : Request) (call : Host.Preparation Native.Term r) (owner address : Nat) : Prop :=
  ∀ role o,o<r.rows*r.stride role → ¬(key (call.output role) o).covers owner address

theorem materialize_frame {M : Type} (ops : Ops M Native.Word Byte) (I : Native.Instructions)
    (S : Native.Schedule) (r : Request) (call : Host.Preparation Native.Term r)
    (source : State Native.Term) (initial : M) (owner address : Nat)
    (outside : Outside r call owner address) :
    ops.observe (materialize ops I S r call source initial) owner address=ops.observe initial owner address := by
  apply bytes_frame
  intro j hj
  exact outside j.1 j.2 ((member r j).mp hj)

/-- Raw little-endian storage, kept separate from numerical instruction ports. -/
structure Codec {M : Type} (ops : Ops M Native.Word Byte) : Prop where
  bytes : ∀ state k n,n<4 → ops.observe state k.owner (k.address+n)=
    (ops.read state k).extractLsb' (8*n) 8

/-- Ordinary pre-launch copy/view binding. These are input word and allocation
frame laws, with no source-success or output-value obligation. In a reuse path,
the returned view addresses the original caller allocation. -/
structure Binding {M : Type} (ops : Ops M Native.Word Byte) (I : Native.Instructions)
    (S : Native.Schedule) (r : Request) (call : Host.Preparation Native.Term r)
    (memory : RegionName → Nat → Native.Word) (before prepared : M) : Prop where
  input : ∀ role b h s c,b<r.batches → h<r.heads role → s<r.sequence → c<r.dimension →
    memory (call.input role).region (r.external role b h s c)=
      ops.read before (key (call.input role) ((call.input role).offset b h s c))
  cosine : ∀ s j,s<r.sequence → j<r.half → memory "C" (s*r.half+j)=
    ops.read before (key call.cosine (s*r.half+j))
  sine : ∀ s j,s<r.sequence → j<r.half → memory "S" (s*r.half+j)=
    ops.read before (key call.sine (s*r.half+j))
  packed : ∀ role o,o<r.rows*r.stride role →
    ops.read prepared (key (call.output role) o)=Native.observe I S call.prepared (region role) o
  frame : ∀ owner address,Outside r call owner address →
    ops.observe prepared owner address=ops.observe before owner address

def invoke {M : Type} (ops : Ops M Native.Word Byte) (I : Native.Instructions)
    (S : Native.Schedule) (r : Request) (call : Host.Preparation Native.Term r) (prepared : M) :=
  if Checked.check r call then
    (Host.invoke Native.algebra r call).map fun result =>
      (result,materialize ops I S r call result.state prepared)
  else none

theorem complete {M : Type} (ops : Ops M Native.Word Byte) (codec : Codec ops)
    (I : Native.Instructions) (S : Native.Schedule) (r : Request)
    (call : Host.Preparation Native.Term r) (memory : RegionName → Nat → Native.Word)
    (before prepared : M) (binding : Binding ops I S r call memory before prepared)
    (entry : call.before=Native.entry memory) (accepted : Checked.check r call=Bool.true) :
    ∃ result final,invoke ops I S r call prepared=some (result,final) ∧
      (∀ role,result.tensor role=call.output role) ∧
      result.cosine=call.cosine ∧ result.sine=call.sine ∧
      (∀ role b h s c,b<r.batches → h<r.heads role → s<r.sequence → c<r.dimension →
        ops.read final (key (result.tensor role) ((result.tensor role).offset b h s c))=
          Native.expectedWord I S r memory (call.input role) role b h s c) ∧
      (∀ role b h s c n,b<r.batches → h<r.heads role → s<r.sequence → c<r.dimension → n<4 →
        ops.observe final (result.tensor role).device
          ((result.tensor role).base+4*((result.tensor role).offset b h s c)+n)=
          (Native.expectedWord I S r memory (call.input role) role b h s c).extractLsb' (8*n) 8) ∧
      (∀ owner address,Outside r call owner address →
        ops.observe final owner address=ops.observe before owner address) ∧
      ABI.Limits r ∧ ABI.PointerLimits r call := by
  obtain ⟨result,run,objects,cosine,sine,metadata,values,frame,coefficients⟩ :=
    Native.whole_native I S r memory call entry
  let final := materialize ops I S r call result.state prepared
  have readback : ∀ role b h s c,b<r.batches → h<r.heads role → s<r.sequence → c<r.dimension →
      ops.read final (key (result.tensor role) ((result.tensor role).offset b h s c))=
        Native.expectedWord I S r memory (call.input role) role b h s c := by
    intro role b h s c hb hh hs hc
    have row := row_bound r b s hb hs
    have inside : r.offset role (b*r.sequence+s) h c<r.rows*r.stride role :=
      lt_of_lt_of_le (offset_window r role (b*r.sequence+s) h c hh hc).2
        (Nat.mul_le_mul_right _ (by omega : b*r.sequence+s+1≤r.rows))
    rw [objects,call.outputAddress role b h s c hb hh hs hc]
    rw [read_materialize ops I S r call result.state prepared role _ inside]
    simpa only [objects,call.outputRegion,call.outputAddress role b h s c hb hh hs hc] using
      values role b h s c hb hh hs hc
  refine ⟨result,final,by simp [invoke,accepted,run,final],objects,cosine,sine,readback,?_,?_,
    (Checked.launch_admitted r call accepted).1,(Checked.launch_admitted r call accepted).2⟩
  · intro role b h s c n hb hh hs hc hn
    change ops.observe final (key (result.tensor role) ((result.tensor role).offset b h s c)).owner
      ((key (result.tensor role) ((result.tensor role).offset b h s c)).address+n)=_
    rw [codec.bytes _ _ _ hn,readback role b h s c hb hh hs hc]
  · intro owner address outside
    exact (materialize_frame ops I S r call result.state prepared owner address outside).trans
      (binding.frame owner address outside)

#axiomsClean read_materialize
#axiomsClean complete
#specNonCircular Binding avoiding [invoke,Host.invoke,Grid.run,Grid.sourceRun]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RopeTransform.Physical
