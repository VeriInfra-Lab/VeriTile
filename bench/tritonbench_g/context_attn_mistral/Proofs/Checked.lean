import bench.tritonbench_g.context_attn_mistral.Proofs.NativeIndex
import bench.tritonbench_g.context_attn_mistral.Proofs.Logical
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealAddress
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.ContextAttnMistral.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open NativeIndex
noncomputable section
set_option maxHeartbeats 3000000

/-- Checked dense launch arguments are synthesized from shapes, so inert
Tensor strides need not equal canonical strides on singleton/empty axes. -/
structure Geometry (c : Host.Call) (tokens : Nat) : Prop where
  heads : 0<c.heads
  kvHeads : 0<c.kvHeads
  group : 0<c.layout.kvGroup
  grouped : c.heads=c.kvHeads*c.layout.kvGroup
  query : c.layout.qRow=c.heads*c.layout.D ∧ c.layout.qHead=c.layout.D ∧ c.layout.qDim=1
  key : c.layout.kRow=c.kvHeads*c.layout.D ∧ c.layout.kHead=c.layout.D ∧ c.layout.kDim=1
  value : c.layout.vRow=c.kvHeads*c.layout.D ∧ c.layout.vHead=c.layout.D ∧ c.layout.vDim=1
  output : c.layout.outRow=c.heads*c.layout.D ∧ c.layout.outHead=c.layout.D ∧ c.layout.outDim=1
  bounds : ∀ b,b<c.batches → c.start b+c.length b≤tokens
  packed : ∀ b z,b<c.batches → z<c.batches → 0<c.length b → 0<c.length z → b≠z →
    c.start b+c.length b≤c.start z ∨ c.start z+c.length z≤c.start b

def headsFor (c : Host.Call) : DataRole → Nat
  | .query | .output => c.heads
  | .key | .value => c.kvHeads

def size (c : Host.Call) (tokens : Nat) (r : DataRole) := tokens*headsFor c r*c.layout.D

def linear (heads dim row head d : Nat) := (row*heads+head)*dim+d

theorem linear_bound (row head d tokens heads dim : Nat)
    (hr : row<tokens) (hh : head<heads) (hd : d<dim) :
    linear heads dim row head d<tokens*heads*dim := by
  have a := Nat.mul_le_mul_right heads (show row+1≤tokens by omega)
  have b := Nat.mul_le_mul_right dim (show row*heads+head+1≤tokens*heads by nlinarith)
  unfold linear
  nlinarith

theorem linear_injective (heads dim r h d r' h' d' : Nat)
    (hh : h<heads) (hh' : h'<heads) (hd : d<dim) (hd' : d'<dim)
    (eq : linear heads dim r h d=linear heads dim r' h' d') : r=r' ∧ h=h' ∧ d=d' := by
  obtain ⟨rows,dims⟩ := VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress.row_column_injective hd hd' eq
  obtain ⟨tokens,heads⟩ := VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress.row_column_injective hh hh' rows
  exact ⟨tokens,heads,dims⟩

theorem head_bound (g : Geometry c tokens) (r : DataRole) (h : Nat) (hh : h<c.heads) :
    r.head c h<headsFor c r := by
  have kv : h/c.layout.kvGroup<c.kvHeads :=
    (Nat.div_lt_iff_lt_mul g.group).mpr (by simpa [←g.grouped] using hh)
  cases r <;> assumption

theorem offset_dense (g : Geometry c tokens) (r : DataRole) (b h row d : Nat) :
    elementOffset c r b h row d=linear (headsFor c r) c.layout.D (c.start b+row) (r.head c h) d := by
  cases r <;> simp only [elementOffset,DataRole.rowStride,DataRole.headStride,DataRole.dimStride,
    DataRole.head,headsFor,g.query.1,g.query.2.1,g.query.2.2,g.key.1,g.key.2.1,g.key.2.2,
    g.value.1,g.value.2.1,g.value.2.2,g.output.1,g.output.2.1,g.output.2.2,Nat.mul_one,linear] <;> ring

theorem coordinate_bound (g : Geometry c tokens) (r : DataRole) (b h row d : Nat)
    (hb : b<c.batches) (hh : h<c.heads) (hr : row<c.length b) (hd : d<c.layout.D) :
    elementOffset c r b h row d<size c tokens r := by
  rw [offset_dense g]
  exact linear_bound _ _ _ _ _ _ (by have := g.bounds b hb;omega) (head_bound g r h hh) hd

theorem packed_unique (g : Geometry c tokens) (b z row row' : Nat)
    (hb : b<c.batches) (hz : z<c.batches) (hr : row<c.length b) (hr' : row'<c.length z)
    (eq : c.start b+row=c.start z+row') : b=z ∧ row=row' := by
  by_cases e : b=z
  · subst z;exact ⟨rfl,by omega⟩
  · have p := g.packed b z hb hz (by omega) (by omega) e
    omega

inductive DType where | fp32 | int32 | other deriving DecidableEq
inductive Role where | query | key | value | output | start | length deriving DecidableEq
structure Tensor where
  object : Nat
  shape : List Nat
  dtype : DType
  device : Nat
  (cuda contiguous negative : Bool)
structure Request where
  tensor : Role → Tensor

def shape (c : Host.Call) (tokens : Nat) : Role → List Nat
  | .query | .output => [tokens,c.heads,c.layout.D]
  | .key | .value => [tokens,c.kvHeads,c.layout.D]
  | .start | .length => [c.batches]

def Schema (c : Host.Call) (tokens : Nat) (r : Request) : Prop := ∀ role,
  (r.tensor role).shape=shape c tokens role ∧
  (r.tensor role).dtype=(match role with | .start | .length => .int32 | _ => .fp32) ∧
  (r.tensor role).device=(r.tensor .query).device ∧
  (r.tensor role).cuda=Bool.true ∧ (r.tensor role).contiguous=Bool.true ∧ (r.tensor role).negative=Bool.false

/-- These are input metadata/storage facts only, not active-address or output
conclusions. The signed-width facts follow from checked nonnegative int32 words. -/
structure Admission (c : Host.Call) (tokens : Nat) (r : Request) (buffers : Buffers) : Prop where
  schema : Schema c tokens r
  geometry : Geometry c tokens
  grid : c.batches<2^31 ∧ c.heads≤65535 ∧ c.blocks≤65535
  strides : c.heads*c.layout.D<2^63 ∧ c.kvHeads*c.layout.D<2^63
  signed : ∀ b,b<c.batches → c.start b<2^31 ∧ c.length b<2^31
  extents : ∀ role,(buffers.data role).extent=size c tokens role
  metaExtent : ∀ role,(buffers.metadataBuffer role).extent=c.batches
  metaWidth : ∀ role,(buffers.metadataBuffer role).width=.int32
  backing : ∀ role,(buffers.data role).base+4*(buffers.data role).extent≤2^64
  metaBacking : ∀ role,(buffers.metadataBuffer role).base+4*c.batches≤2^64
  alignment : ∀ role,0<(buffers.data role).extent → (buffers.data role).base%4=0
  metaAlignment : ∀ role,0<c.batches → (buffers.metadataBuffer role).base%4=0
  separate : ∀ role,role≠.output →
    (buffers.data .output).base+4*(buffers.data .output).extent≤(buffers.data role).base ∨
    (buffers.data role).base+4*(buffers.data role).extent≤(buffers.data .output).base
  metaSeparate : ∀ role,
    (buffers.data .output).base+4*(buffers.data .output).extent≤(buffers.metadataBuffer role).base ∨
    (buffers.metadataBuffer role).base+4*c.batches≤(buffers.data .output).base

def check (c : Host.Call) (tokens : Nat) (r : Request) (buffers : Buffers) : Bool :=
  @decide (Admission c tokens r buffers) (Classical.propDecidable _)
theorem check_iff : check c tokens r buffers=Bool.true ↔ Admission c tokens r buffers := by simp [check]

theorem native_ready (a : Admission c tokens r buffers) : NativeIndex.Ready c .fp32 buffers := by
  have dim : 0<c.layout.D ∧ c.layout.D≤128 := by
    have h := c.channels;simp only [List.mem_cons,List.not_mem_nil,or_false] at h;omega
  have headle : c.layout.kvGroup≤c.heads := by
    have := Nat.mul_le_mul_right c.layout.kvGroup a.geometry.kvHeads
    rw [Nat.one_mul,←a.geometry.grouped] at this
    exact this
  refine ⟨?_,?_,by have := a.grid;omega,fun _=>a.geometry.group,a.geometry.grouped,?_,?_,?_,a.backing,?_,a.separate,?_⟩
  · intro p;have h:=a.grid;exact ⟨lt_trans p.batch.isLt h.1,by have := p.head.isLt;omega,by have := p.block.isLt;omega⟩
  · intro role
    cases role <;> simp only [DataRole.rowStride,DataRole.headStride,DataRole.dimStride,
      a.geometry.query.1,a.geometry.query.2.1,a.geometry.query.2.2,a.geometry.key.1,a.geometry.key.2.1,a.geometry.key.2.2,
      a.geometry.value.1,a.geometry.value.2.1,a.geometry.value.2.2,a.geometry.output.1,a.geometry.output.2.1,a.geometry.output.2.2]
    all_goals exact ⟨by first | exact a.strides.1 | exact a.strides.2,by omega,by decide⟩
  · intro role b hb;rw [a.metaWidth];cases role
    · exact (a.signed b hb).1
    · exact (a.signed b hb).2
  · intro role;rw [a.metaExtent]
  · intro role;rw [a.metaWidth,a.metaExtent];exact a.metaBacking role
  · intro role b h row d hb hh hr hd
    rw [a.extents]
    exact coordinate_bound a.geometry role b h row d hb hh hr hd
  · intro role;rw [a.metaWidth,a.metaExtent];exact a.metaSeparate role

/-- The wrapper exits before launch-only grid, pointer and alias checks. -/
structure EmptyAdmission (c : Host.Call) (tokens : Nat) (request : Request) : Prop where
  empty : c.batches=0 ∨ c.blocks=0
  schema : Schema c tokens request
  geometry : Geometry c tokens
  signed : ∀ b,b<c.batches → c.start b<2^31 ∧ c.length b<2^31

def validate (c : Host.Call) (tokens : Nat) (request : Request) (buffers : Buffers) : Except Unit Bool := by
  classical
  exact if c.batches=0 ∨ c.blocks=0 then
    if EmptyAdmission c tokens request then .ok Bool.false else .error ()
  else if Admission c tokens request buffers then .ok Bool.true else .error ()

theorem validate_launch : validate c tokens request buffers=.ok Bool.true ↔
    (0<c.batches ∧ 0<c.blocks) ∧ Admission c tokens request buffers := by
  classical
  unfold validate
  split_ifs <;> simp_all <;> omega

theorem validate_empty : validate c tokens request buffers=.ok Bool.false ↔ EmptyAdmission c tokens request := by
  classical
  unfold validate
  split_ifs <;> simp_all
  all_goals intro h;have := h.empty;omega

/-- Nonnegativity restores the exact Nat adapter from the physical signed word. -/
theorem signed_word (word : BitVec 32) (nonnegative : 0≤word.toInt) :
    word.toInt=(word.toInt.toNat : Int) ∧ word.toInt.toNat<2^31 := by
  have bound := BitVec.toInt_lt (x:=word)
  norm_num at bound
  exact ⟨(Int.toNat_of_nonneg nonnegative).symm,by omega⟩

/-- Logical role allocations are independent of physical input aliases. -/
structure RoleBinding (c : Host.Call) (tokens : Nat) (A : FlatAlloc) (s : BlockState) : Prop where
  metadata : (∀ b,b<c.batches → (Program.inputs s c.regions).start b=(c.start b : Int)) ∧
    (∀ b,b<c.batches → (Program.inputs s c.regions).sequence b=(c.length b : Int))
  query : A.extent c.regions.query=tokens*c.heads*c.layout.D
  key : A.extent c.regions.key=tokens*c.kvHeads*c.layout.D
  value : A.extent c.regions.value=tokens*c.kvHeads*c.layout.D
  output : A.extent c.regions.out=tokens*c.heads*c.layout.D
  start : c.batches≤A.extent c.regions.start
  length : c.batches≤A.extent c.regions.sequence
  separate : ∀ rg,rg∈[c.regions.query,c.regions.key,c.regions.value,c.regions.start,c.regions.sequence] → c.regions.out≠rg

theorem host_valid (g : Geometry c tokens) (b : RoleBinding c tokens A s) : Host.Valid c A s := by
  refine ⟨b.metadata.1,b.metadata.2,b.start,b.length,g.group,g.grouped,?_,?_,?_,?_,packed_unique g,?_,b.separate⟩
  · intro batch hb h hh row hr d hd;rw [b.query];exact coordinate_bound g .query batch h row d hb hh hr hd
  · intro batch hb h hh row hr d hd;rw [b.key];exact coordinate_bound g .key batch h row d hb hh hr hd
  · intro batch hb h hh row hr d hd;rw [b.value];exact coordinate_bound g .value batch h row d hb hh hr hd
  · intro batch hb h hh row hr d hd;rw [b.output];exact coordinate_bound g .output batch h row d hb hh hr hd
  · intro batch batch' h h' row row' d d' hb hb' hh hh' hr hr' hd hd' eq
    change elementOffset c .output batch h row d=elementOffset c .output batch' h' row' d' at eq
    rw [offset_dense g,offset_dense g] at eq
    exact linear_injective _ _ _ _ _ _ _ _ hh hh' hd hd' eq

#axiomsClean native_ready
#axiomsClean host_valid
#axiomsClean packed_unique
#specNonCircular Admission avoiding [FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ContextAttnMistral.Checked
