import bench.repairs.tritonbench_g.rope_transform.Proofs.Grid
namespace VeriTile.Bench.Repairs.RopeTransform.Host
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

structure Tensor where
  handle : Nat
  allocation : Nat
  region : RegionName
  base : Nat
  allocationBase : Nat
  allocationEnd : Nat
  device : Nat
  shape : List Nat
  strides : List Nat
  dtype : String
  elementBytes : Nat
  strided : Bool
  cuda : Bool
  contiguous : Bool
  negative : Bool
  conjugate : Bool

def Tensor.offset (t : Tensor) (b h s c : Nat) :=
  b*t.strides[0]!+h*t.strides[1]!+s*t.strides[2]!+c*t.strides[3]!
def shape (r : Request) (key : Bool) := [r.batches,r.heads key,r.sequence,r.dimension]
def Outside (r : Request) (name : RegionName) (o : Nat) : Prop :=
  ∀ key, name≠region key ∨ r.rows*r.stride key≤o

/-- Ordinary input validation, transpose/contiguous preparation and return-view
metadata. All value premises concern input copies. Reuse is retained precisely
for empty/singleton transposes. No output formula or successful execution is a
premise; the prepared Q/K cells may alias the original caller inputs. -/
structure Preparation (α : Type) [Inhabited α] (r : Request) where
  before : State α
  prepared : State α
  input : Bool → Tensor
  output : Bool → Tensor
  cosine : Tensor
  sine : Tensor
  inputShape : ∀ key,(input key).shape=shape r key
  outputShape : ∀ key,(output key).shape=shape r key
  cosineShape : cosine.shape=[r.sequence,r.half]
  sineShape : sine.shape=[r.sequence,r.half]
  outputRegion : ∀ key,(output key).region=region key
  cosineRegion : cosine.region="C"
  sineRegion : sine.region="S"
  inputType : ∀ key,(input key).dtype="float32" ∧ (input key).elementBytes=4
  outputType : ∀ key,(output key).dtype="float32" ∧ (output key).elementBytes=4
  coefficientType : cosine.dtype="float32" ∧ sine.dtype="float32" ∧ cosine.elementBytes=4 ∧ sine.elementBytes=4
  devices : ∀ key,(input key).device=(input Bool.false).device ∧ (output key).device=(input Bool.false).device
  coefficientDevices : cosine.device=(input Bool.false).device ∧ sine.device=(input Bool.false).device
  inputAddress : ∀ key b h s c,b<r.batches → h<r.heads key → s<r.sequence → c<r.dimension →
    (input key).offset b h s c=r.external key b h s c
  outputAddress : ∀ key b h s c,b<r.batches → h<r.heads key → s<r.sequence → c<r.dimension →
    (output key).offset b h s c=r.offset key (b*r.sequence+s) h c
  copied : ∀ key b h s c,b<r.batches → h<r.heads key → s<r.sequence → c<r.dimension →
    prepared.mem (region key) (r.offset key (b*r.sequence+s) h c)=
      before.mem (input key).region (r.external key b h s c)
  allocationFrame : ∀ name o,Outside r name o → prepared.mem name o=before.mem name o
  reused : ∀ key,((output key).allocation=(input key).allocation ∧ (output key).base=(input key).base) ↔
    r.rows=0 ∨ r.sequence=1 ∨ r.heads key=1
  reuseRegion : ∀ key,(r.rows=0 ∨ r.sequence=1 ∨ r.heads key=1) → (input key).region=region key
  inputRegionsSeparate : (input Bool.false).region≠(input Bool.true).region ∧
    (∀ key,(input key).region≠"C" ∧ (input key).region≠"S")
  inputApart : ∀ key target, key≠target → (input key).allocation≠(output target).allocation
  freshApart : ∀ key, ¬(r.rows=0 ∨ r.sequence=1 ∨ r.heads key=1) →
    (input key).region≠"Q" ∧ (input key).region≠"K"
  inputFits : ∀ key,0<r.rows*r.stride key → (input key).allocationBase≤(input key).base ∧
    (input key).base+4*(r.rows*r.stride key)≤(input key).allocationEnd ∧ (input key).allocationEnd≤2^64
  outputFits : ∀ key,0<r.rows*r.stride key → (output key).allocationBase≤(output key).base ∧
    (output key).base+4*(r.rows*r.stride key)≤(output key).allocationEnd ∧ (output key).allocationEnd≤2^64
  coefficientFits : 0<r.sequence*r.half → cosine.base+4*(r.sequence*r.half)≤cosine.allocationEnd ∧
    sine.base+4*(r.sequence*r.half)≤sine.allocationEnd ∧ cosine.allocationEnd≤2^64 ∧ sine.allocationEnd≤2^64
  outputApart : ∀ key target,key≠target →
    (output key).base+4*(r.rows*r.stride key)≤(output target).base ∨
    (output target).base+4*(r.rows*r.stride target)≤(output key).base
  coefficientApart : ∀ key,
    ((output key).base+4*(r.rows*r.stride key)≤cosine.base ∨ cosine.base+4*(r.sequence*r.half)≤(output key).base) ∧
    ((output key).base+4*(r.rows*r.stride key)≤sine.base ∨ sine.base+4*(r.sequence*r.half)≤(output key).base)

structure Result (α : Type) where
  q : Tensor
  k : Tensor
  cosine : Tensor
  sine : Tensor
  state : State α

def invoke {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (call : Preparation α r) : Option (Result α) :=
  (Grid.sourceRun M r call.prepared r.rows).map fun s =>
    ⟨call.output Bool.false,call.output Bool.true,call.cosine,call.sine,s⟩
def Result.tensor (result : Result α) (key : Bool) := if key then result.k else result.q

/-- Independent full logical formula, including every channel in the last
dimension. The coefficient tensors are ordinary caller inputs. -/
def expected {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (before : State α)
    (input : Tensor) (key : Bool) (b h s c : Nat) : α :=
  let j := if c<r.half then c else c-r.half
  scalar M ((before.mem input.region (r.external key b h s j)).read .fp32)
    ((before.mem input.region (r.external key b h s (j+r.half))).read .fp32)
    ((before.mem "C" (s*r.half+j)).read .fp32)
    ((before.mem "S" (s*r.half+j)).read .fp32) (decide (r.half≤c))

theorem current_to_input {α : Type} [Inhabited α] (M : Algebra α) (r : Request)
    (call : Preparation α r) (key second : Bool) (b h s j : Nat)
    (hb : b<r.batches) (hh : h<r.heads key) (hs : s<r.sequence) (hj : j<r.half) :
    value M r call.prepared key (b*r.sequence+s) h second j=
      expected M r call.before (call.input key) key b h s (r.column second j) := by
  have h0 : j<r.dimension := by unfold Request.dimension; omega
  have h1 : j+r.half<r.dimension := by unfold Request.dimension; omega
  have hc : ∀ o,call.prepared.mem "C" o=call.before.mem "C" o := by
    intro o; apply call.allocationFrame; intro key; left; cases key <;> decide
  have hsn : ∀ o,call.prepared.mem "S" o=call.before.mem "S" o := by
    intro o; apply call.allocationFrame; intro key; left; cases key <;> decide
  unfold value
  rw [call.copied key b h s j hb hh hs h0,call.copied key b h s (j+r.half) hb hh hs h1,hc,hsn]
  have hm : (b*r.sequence+s)%r.sequence=s := by simp [Nat.add_mod,Nat.mod_eq_of_lt hs]
  rw [hm]
  cases second
  · simp [expected,Request.column,hj,show ¬ r.half≤j by omega]
  · have hn : ¬ j+r.half<r.half := by omega
    simp [expected,Request.column,hn,Nat.add_sub_cancel_right]

theorem column_coverage (r : Request) (c : Nat) (hc : c<r.dimension) :
    ∃ second j,j<r.half ∧ r.column second j=c := by
  by_cases h : c<r.half
  · exact ⟨Bool.false,c,h,by simp [Request.column]⟩
  · refine ⟨Bool.true,c-r.half,?_,?_⟩
    · unfold Request.dimension at hc; omega
    · simp [Request.column]; omega

/-- Successful complete source launch, all four returned objects, every logical
Q/K output, coefficient identity/preservation and exact logical-word frame. -/
theorem whole_typed {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (call : Preparation α r) :
    ∃ result,invoke M r call=some result ∧
      (∀ key,result.tensor key=call.output key) ∧
      result.cosine=call.cosine ∧ result.sine=call.sine ∧
      (∀ key,(result.tensor key).shape=shape r key ∧ (result.tensor key).dtype="float32" ∧
        (result.tensor key).device=(call.input Bool.false).device) ∧
      (∀ key b h s c,b<r.batches → h<r.heads key → s<r.sequence → c<r.dimension →
        result.state.mem (result.tensor key).region ((result.tensor key).offset b h s c)=
          Cell.mk .fp32 (expected M r call.before (call.input key) key b h s c)) ∧
      (∀ name o,Outside r name o → result.state.mem name o=call.before.mem name o) ∧
      (∀ o,result.state.mem "C" o=call.before.mem "C" o ∧ result.state.mem "S" o=call.before.mem "S" o) := by
  obtain ⟨he,hv,hf⟩ := Grid.complete M r call.prepared
  let result : Result α := ⟨call.output Bool.false,call.output Bool.true,call.cosine,call.sine,
    Grid.run M r call.prepared r.rows⟩
  have ht : ∀ key,result.tensor key=call.output key := by intro key; cases key <;> rfl
  have frameAll : ∀ name o,Outside r name o → result.state.mem name o=call.before.mem name o :=
    fun name o ho => (hf name o ho).trans (call.allocationFrame name o ho)
  refine ⟨result,by simp [invoke,he,result],ht,rfl,rfl,?_,?_,frameAll,?_⟩
  · intro key
    rw [ht]
    exact ⟨call.outputShape key,(call.outputType key).1,(call.devices key).2⟩
  · intro key b h s c hb hh hs hc
    rw [ht,call.outputRegion,call.outputAddress key b h s c hb hh hs hc]
    obtain ⟨second,j,hj,hec⟩ := column_coverage r c hc
    rw [←hec]
    have hrow := hv key (b*r.sequence+s) h second j (row_bound r b s hb hs) hh hj
    rw [current_to_input M r call key second b h s j hb hh hs hj] at hrow
    exact hrow
  · intro o
    constructor
    · apply frameAll; intro key; left; cases key <;> decide
    · apply frameAll; intro key; left; cases key <;> decide

theorem empty_return {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (call : Preparation α r)
    (empty : r.rows=0) :
    invoke M r call=some ⟨call.output Bool.false,call.output Bool.true,call.cosine,call.sine,call.prepared⟩ := by
  simp [invoke,empty,Grid.sourceRun]

#axiomsClean whole_typed
#axiomsClean empty_return
#axiomsClean column_coverage
#specNonCircular Preparation avoiding [invoke,Grid.run,Grid.sourceRun,FP.Structural.exec]
#specNonCircular expected avoiding [invoke,Grid.run,Grid.sourceRun,FP.Structural.exec,value,T.laneResult]
end
end VeriTile.Bench.Repairs.RopeTransform.Host
