import bench.tritonbench_g.diag_ssm_triton.Proofs.ByteMemory
import bench.tritonbench_g.diag_ssm_triton.Proofs.WordCodec
import bench.tritonbench_g.cross_entropy1.Proofs.TensorModel

namespace VeriTile.Bench.Repairs.KldivOps.Storage
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (coordinateOffset)
noncomputable section

/-- A real Torch view, with its original identity, offset, strides and lazy
negative bit. Conjugate resolution is the identity on selected FP32 values. -/
structure Tensor where
  objectId : Nat
  shape : List Nat
  strides : List Nat
  offset : Nat
  backing : Backing
  negative : Bool
  device : Nat
  rank : strides.length=shape.length
  capacity : ∀ i, i<shape.prod →
    4*(offset+coordinateOffset shape strides i+1)≤backing.bytes

def Tensor.numel (t : Tensor) := t.shape.prod
def Tensor.index (t : Tensor) (i : Nat) := t.offset+coordinateOffset t.shape t.strides i
def Tensor.address (t : Tensor) (i : Nat) := t.backing.base+4*t.index i

def value {Word : Type} (codec : Codec Word) (neg : Word→Word)
    (m : Heap) (t : Tensor) (i : Nat) : Word :=
  let word := load codec m (t.address i)
  if t.negative then neg word else word

theorem byte_inside (t : Tensor) (i : Nat) (hi : i<t.numel) (k : Fin 4) :
    t.backing.contains (t.address i+k.val) := by
  have h := t.capacity i hi
  simp only [Tensor.address,Tensor.index,Backing.contains]
  omega

def Written (t : Tensor) (p : Nat) : Prop :=
  ∃ i, i<t.numel ∧ t.address i≤p ∧ p<t.address i+4

def OutsideFrame (before after : Heap) (t : Tensor) : Prop :=
  ∀ p, ¬Written t p → after p=before p

theorem outsideFrame_refl (m : Heap) (t : Tensor) : OutsideFrame m m t := by intro _ _; rfl

theorem outsideFrame_trans {a b c : Heap} {t : Tensor}
    (ab : OutsideFrame a b t) (bc : OutsideFrame b c t) : OutsideFrame a c t := by
  intro p hp; rw [bc p hp,ab p hp]

theorem preserves_value {Word : Type} (codec : Codec Word) (neg : Word→Word)
    (a b : Heap) (t : Tensor) (frame : Frame a b t.backing) (i : Nat) (hi : i<t.numel) :
    value codec neg b t i=value codec neg a t i := by
  unfold value load
  have h : (fun k : Fin 4 => b (t.address i+k.val))=(fun k : Fin 4 => a (t.address i+k.val)) := by
    funext k; exact frame _ (byte_inside t i hi k)
  rw [h]

/-- Same-call dense views. Inputs may alias arbitrarily; only fresh work must
be disjoint from each input. This is a relation over one actual heap, never a
universal load law for arbitrary unrelated abstract worlds. -/
structure Arena where
  prediction : View
  target : View
  output : View
  predApart : output.backing.disjoint prediction.backing
  targetApart : output.backing.disjoint target.backing

def represented {Word : Type} [Inhabited Word] (codec : Codec Word) (a : Arena) (m : Heap) : State Word where
  mem := fun r i => .mk .real (load codec m
    ((if r="Y" then a.prediction else if r="T" then a.target else a.output).address i))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

theorem represented_prediction {Word : Type} [Inhabited Word] (codec : Codec Word) (a : Arena) (m : Heap) (i : Nat) :
    ((represented codec a m).mem "Y" i).read .real=load codec m (a.prediction.address i) := by
  simp [represented]

theorem represented_target {Word : Type} [Inhabited Word] (codec : Codec Word) (a : Arena) (m : Heap) (i : Nat) :
    ((represented codec a m).mem "T" i).read .real=load codec m (a.target.address i) := by
  simp [represented]

/-- Ordinary ordered store realization, applied to the values from the actual
successful typed program state. It does not contain a KLDIV output formula. -/
def commit {Word : Type} [Inhabited Word] (codec : Codec Word) (a : Arena)
    (state : State Word) (m : Heap) : Heap :=
  writeValues codec a.output (List.range a.output.words)
    (fun i => (state.mem "O" i).read .real) m

theorem commit_read {Word : Type} [Inhabited Word] (codec : Codec Word) (a : Arena)
    (state : State Word) (m : Heap) (i : Nat) (hi : i<a.output.words) :
    load codec (commit codec a state m) (a.output.address i)=(state.mem "O" i).read .real := by
  apply writeValues_read
  exact List.mem_range.mpr hi

theorem commit_caller_frame {Word : Type} [Inhabited Word] (codec : Codec Word) (a : Arena)
    (state : State Word) (m : Heap) (caller : Backing) (apart : a.output.backing.disjoint caller) :
    Frame m (commit codec a state m) caller := by
  apply writeValues_other_backing _ _ _ _ _ _ apart
  intro i hi; exact List.mem_range.mp hi

theorem commit_padding {Word : Type} [Inhabited Word] (codec : Codec Word) (a : Arena)
    (state : State Word) (m : Heap) (p : Nat)
    (outside : ∀ i, i<a.output.words → p<a.output.address i ∨ a.output.address i+4≤p) :
    commit codec a state m p=m p := by
  apply writeValues_frame
  intro i hi; exact outside i (List.mem_range.mp hi)

/-- A concrete separated nonempty arena proves that the physical interface
has a model. It admits arbitrary byte heaps and arbitrary input aliasing. -/
def witness : Arena where
  prediction := ⟨⟨4,12,by norm_num⟩,0,3,by norm_num⟩
  target := ⟨⟨4,12,by norm_num⟩,0,3,by norm_num⟩
  output := ⟨⟨32,12,by norm_num⟩,0,3,by norm_num⟩
  predApart := by right; norm_num [Backing.disjoint]
  targetApart := by right; norm_num [Backing.disjoint]

theorem representation_inhabited {Word : Type} [Inhabited Word] (codec : Codec Word) (m : Heap) :
    ∃ state : State Word, ∀ i,
      (state.mem "Y" i).read .real=load codec m (witness.prediction.address i) ∧
      (state.mem "T" i).read .real=load codec m (witness.target.address i) := by
  exact ⟨represented codec witness m,fun i =>
    ⟨represented_prediction codec witness m i,represented_target codec witness m i⟩⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Storage
