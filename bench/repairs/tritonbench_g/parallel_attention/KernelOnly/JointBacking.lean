import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Contract

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.CommonBacking
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open TensorBinding
noncomputable section
set_option maxHeartbeats 500000
variable {κ : Type} {regions : κ → RegionName} {shapes : κ → List Nat}
  {read : Memory} {source : State Word}

structure Slot (κ : Type) where
  port : κ
  index : Nat

def Slot.active (views : κ → View) (writes : RegionName → Nat → Prop) (slot : Slot κ) : Prop :=
  slot.index<(views slot.port).tensor.metadata.numel ∧ writes (views slot.port).region slot.index
def Slot.address (views : κ → View) (slot : Slot κ) : Nat := CommonBacking.address (views slot.port) slot.index

def PhysicalWrites (views : κ → View) (writes : RegionName → Nat → Prop) (a : Nat) : Prop :=
  ∃ slot : Slot κ,slot.active views writes ∧ a=slot.address views

/-- One source-derived physical post for every output port simultaneously.
Only actual source-written words change. The chosen slot depends solely on
integer addresses and the source footprint; its value is read from the
proven final tagged state, never supplied as a desired output. -/
def jointPost (read : Memory) (views : κ → View) (writes : RegionName → Nat → Prop)
    (final : State Word) (a : Nat) : VeriTile.Torch.FP32.Word := by
  classical
  exact if h : PhysicalWrites views writes a then
    ⟨(final.mem (views h.choose.port).region h.choose.index).read .real⟩ else read a

theorem Outputs.address_unique (b : Outputs κ regions shapes read source)
    (x y : Slot κ) (hx : x.index<(b.view x.port).tensor.metadata.numel)
    (hy : y.index<(b.view y.port).tensor.metadata.numel)
    (same : x.address b.view=y.address b.view) : x=y := by
  have ports : x.port=y.port := by
    by_contra unequal
    exact b.separate x.port y.port unequal x.index y.index hx hy same
  have indices : x.index=y.index := by
    change address (b.view x.port) x.index=address (b.view y.port) y.index at same
    rw [ports] at same
    unfold address at same
    omega
  cases x; cases y
  simp_all

theorem jointPost_at (b : Outputs κ regions shapes read source) (writes : RegionName → Nat → Prop)
    (final : State Word) (k : κ) (i : Nat) (inside : i<(b.view k).tensor.metadata.numel)
    (written : writes (b.view k).region i) :
    (jointPost read b.view writes final (address (b.view k) i)).bits=
      (final.mem (b.view k).region i).read .real := by
  classical
  unfold jointPost
  have h : PhysicalWrites b.view writes (address (b.view k) i) := ⟨⟨k,i⟩,⟨inside,written⟩,rfl⟩
  rw [dif_pos h]
  have same : h.choose=⟨k,i⟩ := b.address_unique h.choose ⟨k,i⟩ h.choose_spec.1.1 inside h.choose_spec.2.symm
  rw [same]

theorem jointPost_frame (read : Memory) (views : κ → View) (writes : RegionName → Nat → Prop)
    (final : State Word) (a : Nat) (outside : ¬PhysicalWrites views writes a) :
    jointPost read views writes final a=read a := by
  classical
  simp only [jointPost,dif_neg outside]

/-- All output slots, including inactive slots, bind to this same post.
Unwritten words use the initial backing equality and the source frame. -/
theorem jointPost_binding (b : Outputs κ regions shapes read source) (writes : RegionName → Nat → Prop)
    (final : State Word) (frame : ∀ R i,¬writes R i → final.mem R i=source.mem R i)
    (k : κ) (i : Nat) (inside : i<(b.view k).tensor.metadata.numel) :
    (jointPost read b.view writes final (address (b.view k) i)).bits=
      (final.mem (b.view k).region i).read .real := by
  classical
  by_cases written : writes (b.view k).region i
  · exact jointPost_at b writes final k i inside written
  · have outside : ¬PhysicalWrites b.view writes (address (b.view k) i) := by
      rintro ⟨slot,active,same⟩
      have eq : slot=⟨k,i⟩ := b.address_unique slot ⟨k,i⟩ active.1 inside same.symm
      subst slot
      exact written active.2
    rw [jointPost_frame read b.view writes final _ outside,frame _ _ written,b.initial k i inside]

/-- The actual Tensor logical API reads its word from the one common
physical memory, including the actual storage offset. -/
theorem logical_word (read : Memory) (v : View) (i : Nat) (inside : i<v.tensor.metadata.numel) :
    (VeriTile.Torch.FP32.logicalValue read v.tensor v.laws i).bits=(read (address v i)).bits := by
  rw [← VeriTile.Torch.FP32.direct_read_eq_logical read v.tensor v.laws v.contiguous v.noNeg i inside]
  have base := pointer_base v (by omega)
  change (read (_+4*i)).bits=_
  rw [← base]
  rfl

theorem jointPost_logical (b : Outputs κ regions shapes read source) (writes : RegionName → Nat → Prop)
    (final : State Word) (frame : ∀ R i,¬writes R i → final.mem R i=source.mem R i)
    (k : κ) (i : Nat) (inside : i<(b.view k).tensor.metadata.numel) :
    (VeriTile.Torch.FP32.logicalValue (jointPost read b.view writes final)
      (b.view k).tensor (b.view k).laws i).bits=(final.mem (b.view k).region i).read .real := by
  rw [logical_word _ _ i inside]
  exact jointPost_binding b writes final frame k i inside

theorem jointPost_input_storage (read : Memory) (views : κ → View) (ins : List View)
    (separate : InputSeparation views ins) (writes : RegionName → Nat → Prop) (final : State Word)
    (v : View) (member : v∈ins) (a : Nat) (inside : inStorage v a) :
    jointPost read views writes final a=read a := by
  apply jointPost_frame
  rintro ⟨slot,active,rfl⟩
  exact separate slot.port slot.index active.1 v member inside

theorem address_in_storage (v : View) (i : Nat) (inside : i<v.tensor.metadata.numel) :
    inStorage v (address v i) := by
  have position := v.laws.contiguous v.contiguous i inside
  have bound := v.laws.inBounds i inside
  have base := pointer_base v (by omega)
  have positive : v.tensor.metadata.numel≠0 := by omega
  simp only [address,base,VeriTile.Torch.FP32.dataView,VeriTile.Torch.FP32.DataView.base,
    positive,ite_false,VeriTile.Torch.FP32.Storage.address,inStorage] at *
  omega

theorem jointPost_input_logical (read : Memory) (views : κ → View) (ins : List View)
    (separate : InputSeparation views ins) (writes : RegionName → Nat → Prop) (final : State Word)
    (v : View) (member : v∈ins) (i : Nat) (inside : i<v.tensor.metadata.numel) :
    VeriTile.Torch.FP32.logicalValue (jointPost read views writes final) v.tensor v.laws i=
      VeriTile.Torch.FP32.logicalValue read v.tensor v.laws i := by
  have words : (VeriTile.Torch.FP32.logicalValue (jointPost read views writes final) v.tensor v.laws i).bits=
      (VeriTile.Torch.FP32.logicalValue read v.tensor v.laws i).bits := by
    rw [logical_word _ _ _ inside,logical_word _ _ _ inside,
      jointPost_input_storage read views ins separate writes final v member _ (address_in_storage v i inside)]
  exact congrArg (fun bits : Word => (⟨bits⟩ : VeriTile.Torch.FP32.Word)) words

/-- A single common before/after backing relation for all ports. The
logical observations, input storage frame, and union frame all use exactly
the same `post` reader. -/
structure JointBinding (read post : Memory) (source final : State Word) (views : κ → View)
    (ins : List View) (writes : RegionName → Nat → Prop) : Prop where
  beforeOutputs : ∀ k i,i<(views k).tensor.metadata.numel →
    (source.mem (views k).region i).read .real=(read (address (views k) i)).bits
  beforeInputs : InputBinding read source ins
  afterOutputs : ∀ k i,i<(views k).tensor.metadata.numel →
    (post (address (views k) i)).bits=(final.mem (views k).region i).read .real
  afterInputs : InputBinding post final ins
  logicalOutputs : ∀ k i,i<(views k).tensor.metadata.numel →
    (VeriTile.Torch.FP32.logicalValue post (views k).tensor (views k).laws i).bits=
      (final.mem (views k).region i).read .real
  logicalInputs : ∀ v∈ins,∀ i,i<v.tensor.metadata.numel →
    VeriTile.Torch.FP32.logicalValue post v.tensor v.laws i=VeriTile.Torch.FP32.logicalValue read v.tensor v.laws i
  inputStorageFrame : ∀ v∈ins,∀ a,inStorage v a → post a=read a
  frame : ∀ a,¬PhysicalWrites views writes a → post a=read a

theorem realize_joint (b : Outputs κ regions shapes read source) (ins : List View)
    (initialInputs : InputBinding read source ins) (separate : InputSeparation b.view ins)
    (writes : RegionName → Nat → Prop) (final : State Word)
    (sourceFrame : ∀ R i,¬writes R i → final.mem R i=source.mem R i)
    (inputFrame : ∀ v∈ins,final.mem v.region=source.mem v.region) :
    JointBinding read (jointPost read b.view writes final) source final b.view ins writes := by
  refine ⟨b.initial,initialInputs,jointPost_binding b writes final sourceFrame,?_,
    jointPost_logical b writes final sourceFrame,?_,?_,jointPost_frame read b.view writes final⟩
  · intro v member i inside
    rw [inputFrame v member,initialInputs v member i inside,
      jointPost_input_storage read b.view ins separate writes final v member _ (address_in_storage v i inside)]
  · intro v member i inside
    exact jointPost_input_logical read b.view ins separate writes final v member i inside
  · intro v member a inside
    exact jointPost_input_storage read b.view ins separate writes final v member a inside

#axiomsClean Outputs.address_unique
#axiomsClean jointPost_at
#axiomsClean jointPost_frame
#axiomsClean jointPost_binding
#axiomsClean logical_word
#axiomsClean jointPost_logical
#axiomsClean jointPost_input_storage
#axiomsClean address_in_storage
#axiomsClean jointPost_input_logical
#axiomsClean realize_joint
#specNonCircular PhysicalWrites avoiding [FP.Structural.exec,runPrograms,NativeStage.forwardRun,NativeStage.combinedRun]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.CommonBacking
