import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelGeometry
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Native

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open KernelTensor KernelGrid
noncomputable section
open Classical
set_option maxHeartbeats 5000000

abbrev Node := Native.Node
abbrev Instructions := Native.Instructions
abbrev model := SoftmaxTriton1.NativeGraph.algebra
abbrev owner := SoftmaxTriton1.NativeGraph.owner

def wordRead (before : NativeCells Word) (address : Nat) : Word := (before address).read .fp32

def lift (before : NativeCells Word) : NativeCells Node :=
  fun address => Native.entry (fun _ i => before i) "native" address

def nodeRead (before : NativeCells Word) (address : Nat) : Node :=
  SoftmaxTriton1.NativeGraph.atom (wordRead before address)

@[simp] theorem model_profile (I : Instructions) (V : Nat) :
    FP.ProfiledRewrite.engine (model I V)=model I V := rfl

@[simp] theorem lift_read (before : NativeCells Word) (address : Nat) :
    ((lift before) address).read .fp32=nodeRead before address := by
  exact Native.read_fp32 (fun _ i => before i) "native" address

theorem word_binding (c : TensorCall) (before : NativeCells Word) :
    InputBinding c (wordRead before) before := by intro h hh t ht; rfl

theorem node_binding (c : TensorCall) (before : NativeCells Word) :
    InputBinding c (nodeRead before) (lift before) := by
  intro h hh t ht
  exact lift_read before _

theorem view_entry (c : TensorCall) (before : NativeCells Word) :
    view c (nodeRead before) (wordRead before) (lift before)=
      Native.entry (view c (wordRead before) (wordRead before) before) := by
  funext r i
  by_cases a : r="input"
  · subst r; rfl
  by_cases b : r="start"
  · subst r; rfl
  by_cases d : r="length"
  · subst r; rfl
  by_cases e : r="output"
  · subst r; rfl
  have left : view c (nodeRead before) (wordRead before) (lift before) r i=.mk .nat 0 := by
    simp only [view,if_neg a,if_neg b,if_neg d,if_neg e]
  have right : view c (wordRead before) (wordRead before) before r i=.mk .nat 0 := by
    simp only [view,if_neg a,if_neg b,if_neg d,if_neg e]
  let convert : Cell Word → Cell Node := fun cell =>
    match cell with | .mk dtype value => .mk dtype (Native.inject dtype value)
  change view c (nodeRead before) (wordRead before) (lift before) r i=
    convert (view c (wordRead before) (wordRead before) before r i)
  rw [left,right]
  rfl

theorem events_entry (c : TensorCall) (before : NativeCells Word) (work : List Grid.PID) :
    Grid.events (params c) (view c (nodeRead before) (wordRead before) (lift before)) work=
      Grid.events (params c) (view c (wordRead before) (wordRead before) before) work := by
  simp [Grid.events,Grid.active,count,Grid.state,params,view]

/-- Ordered fixed-instruction graph, applied to original logical Tensor words.
All scalar instruction meanings and the selected layout V remain explicit. -/
def reference (I : Instructions) (V : Nat) (c : TensorCall) (before : NativeCells Word)
    (b head n : Nat) : Word :=
  SoftmaxTriton1.NativeGraph.row I V (length c (wordRead before) b) (params c).width
    (fun j => (VeriTile.Torch.FP32.logicalValue (fun address => ⟨wordRead before address⟩)
      c.input.tensor c.input.laws (head*cols c.input.observation+(start c (wordRead before) b+j))).bits) n

theorem row_congr (I : Instructions) (V n B : Nat) (xs ys : Nat → Word) (i : Nat)
    (same : ∀ j,j < n → xs j=ys j) :
    SoftmaxTriton1.NativeGraph.row I V n B xs i=SoftmaxTriton1.NativeGraph.row I V n B ys i := by
  have masked : SoftmaxTriton1.NativeGraph.masked n xs=SoftmaxTriton1.NativeGraph.masked n ys := by
    funext j
    by_cases inside : j < n <;> simp [SoftmaxTriton1.NativeGraph.masked,inside,same j]
  simp only [SoftmaxTriton1.NativeGraph.row,SoftmaxTriton1.NativeGraph.denominator,
    SoftmaxTriton1.NativeGraph.numerator,SoftmaxTriton1.NativeGraph.maximum,masked]

theorem target_logical (I : Instructions) (V : Nat) (c : TensorCall) (before : NativeCells Word)
    (contract : InputContract c.observation (observedStart c (wordRead before)) (observedLength c (wordRead before)))
    (launch : VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation)
    (b head n : Nat) (hb : b < rows c.start.observation) (hh : head < rows c.input.observation) :
    Native.target I V (params c) (view c (wordRead before) (wordRead before) before) (b,head) n=
      reference I V c before b head n := by
  unfold Native.target reference
  rw [count_view]
  apply row_congr
  intro j hj
  have links := KernelBinding.source_native_load c (wordRead before) (wordRead before) before
    (word_binding c before) contract launch b head j hb hh hj
  have cells : Native.rowInput (params c) (view c (wordRead before) (wordRead before) before) (b,head) j=
      logicalValue (wordRead before) c.input head (start c (wordRead before) b+j) := by
    simpa [Native.rowInput,Grid.state,params,startIndex,view,start,observedStart] using links.1.trans links.2.1
  have normal : c.input.tensor.metadata.isNeg=Bool.false :=
    contract.metadata.tensors.2.2.2.2.2.2.2.2.2.2.2 c.input.observation
      (by simp [tensors,TensorCall.observation])
  exact cells.trans (KernelBinding.logical_fp32 (wordRead before) c.input normal head
    (start c (wordRead before) b+j))

/-- A physical native cell selects the owning lane of its unique active store. -/
def Selected (c : TensorCall) (before : NativeCells Word) (work : List Grid.PID) (address : Nat) : Prop :=
  ∃ e∈Grid.events (params c) (view c (wordRead before) (wordRead before) before) work,
    address=c.output.tensor.dataPtr.toNat+
      4*Grid.offset (params c) (view c (wordRead before) (wordRead before) before) e.1 e.2

def observe (V : Nat) (c : TensorCall) (before : NativeCells Word) (work : List Grid.PID)
    (after : Grid.Memory Node) : NativeCells Word := fun address =>
  if found : Selected c before work address then
    let e := Classical.choose found
    .mk .fp32 (((after "output" (Grid.offset (params c)
      (view c (wordRead before) (wordRead before) before) e.1 e.2)).read .fp32)
        (owner V (params c).width e.2.1.val))
  else before address

theorem observe_readback (V : Nat) (c : TensorCall) (before : NativeCells Word) (work : List Grid.PID)
    (after : Grid.Memory Node)
    (unique : Grid.Unique (params c) (view c (wordRead before) (wordRead before) before) work)
    (e : Grid.Event (params c))
    (live : e∈Grid.events (params c) (view c (wordRead before) (wordRead before) before) work) :
    observe V c before work after (c.output.tensor.dataPtr.toNat+
      4*Grid.offset (params c) (view c (wordRead before) (wordRead before) before) e.1 e.2)=
      .mk .fp32 (((after "output" (Grid.offset (params c)
        (view c (wordRead before) (wordRead before) before) e.1 e.2)).read .fp32)
          (owner V (params c).width e.2.1.val)) := by
  have found : Selected c before work (c.output.tensor.dataPtr.toNat+
      4*Grid.offset (params c) (view c (wordRead before) (wordRead before) before) e.1 e.2) := ⟨e,live,rfl⟩
  have spec := Classical.choose_spec found
  have chosen : Classical.choose found=e := unique _ spec.1 e live (by omega)
  rw [observe,dif_pos found]
  simp only [chosen]

theorem observe_frame (V : Nat) (c : TensorCall) (before : NativeCells Word) (work : List Grid.PID)
    (after : Grid.Memory Node)
    (order : (Grid.fullGrid (rows c.start.observation) (rows c.input.observation)).Perm work)
    (address : Nat) (outside : Exterior c (wordRead before) address) :
    observe V c before work after address=before address := by
  have miss : ¬Selected c before work address := by
    rintro ⟨e,he,equal⟩
    obtain ⟨event,active⟩ := (Grid.mem_events _ _ _ _).mp he
    have legal := (Grid.mem_fullGrid _ _ _).mp (order.mem_iff.mpr event)
    have count : e.2.1.val < length c (wordRead before) e.1.1 := by simpa [Grid.active] using active
    apply outside e.1.1 legal.1 e.1.2 legal.2 e.2.1.val count
    simpa [outputAddress] using equal
  simp [observe,miss]

#axiomsClean target_logical
#axiomsClean observe_readback
#axiomsClean observe_frame
#specNonCircular reference avoiding [source,Grid.run,observe,outputValues]
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelNative
