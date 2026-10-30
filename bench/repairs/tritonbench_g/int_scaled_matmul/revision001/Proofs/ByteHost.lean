import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.ByteExecution

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ByteHost
open VeriTile.Triton FP.Structural
open Contract TensorBinding InputAdapter FiniteExecution FiniteIndex OutputBytes
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 6000000

def run (scaled : Bool) (a b c scale : TensorBinding.Tensor) (p : Contract.Config)
    (suffix : State Word → Suffix (launch a b p)) (t : ByteExecution.Machine) :=
  if checkTensorInputs scaled a.metadata b.metadata c.metadata scale.metadata p=.ok () then
    if Empty a.metadata b.metadata then some (c,t)
    else (ByteExecution.sequence (ByteExecution.one (launch a b p) c scaled suffix)
      (List.range (count (rows a.metadata) (columns b.metadata) p)) t).map (fun final => (c,final))
  else none

def logicalRun (scaled : Bool) (a b c scale : TensorBinding.Tensor) (p : Contract.Config)
    (suffix : State Word → Suffix (launch a b p)) (s : State Word) :=
  if checkTensorInputs scaled a.metadata b.metadata c.metadata scale.metadata p=.ok () then
    if Empty a.metadata b.metadata then some (c,s)
    else (GridExecution.sequence (ByteExecution.logicalOne (launch a b p) scaled suffix)
      (List.range (count (rows a.metadata) (columns b.metadata) p)) s).map (fun final => (c,final))
  else none

theorem size_eq (scaled : Bool) (a b c scale : TensorBinding.Tensor) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p) :
    rows a.metadata*columns b.metadata=c.metadata.numel := by
  simp [Torch.TensorMetadata.numel,h.shapes.2.1]

theorem run_render (scaled : Bool) (a b c scale : TensorBinding.Tensor) (p : Contract.Config)
    (suffix : State Word → Suffix (launch a b p))
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (read : Read) (s : State Word) (bound : Bound read c s) :
    run scaled a b c scale p suffix (s,read)=
      (logicalRun scaled a b c scale p suffix s).map
        (fun result => (result.1,(result.2,render read c result.2))) := by
  have checked := (checkTensorInputs_iff scaled a.metadata b.metadata c.metadata scale.metadata p).mpr h
  by_cases empty : Empty a.metadata b.metadata
  · simp only [run,logicalRun,if_pos checked,if_pos empty,Option.map_some,render_initial read c s bound]
  · have env := (execution_of_nonempty scaled a.metadata b.metadata c.metadata scale.metadata p h empty).1
    have seq := ByteExecution.sequence_render (launch a b p) c scaled suffix h.config env rfl
      (size_eq scaled a b c scale p h) read (List.range (count (rows a.metadata) (columns b.metadata) p))
      (fun _ hi => List.mem_range.mp hi) s
    rw [render_initial read c s bound] at seq
    simp only [run,logicalRun,if_pos checked,if_neg empty,seq,Option.map_map]
    rfl

def call (scaled : Bool) (cfg : FP.Config) (u : Nat → Word → Int)
    (a b c scale : TensorBinding.Tensor) (p : Contract.Config) (t : ByteExecution.Machine) :=
  run scaled a b c scale p (fun s => if scaled then scaledSuffix cfg u (launch a b p) scales s
    else fun carry i => carry i) t

def nativeCall (native : CheckedNative.Instructions) (a b c scale : TensorBinding.Tensor)
    (p : Contract.Config) (t : ByteExecution.Machine) :=
  run Bool.true a b c scale p (fun s => CheckedNative.suffix (CheckedNative.ports native) (launch a b p) scales s) t

theorem call_render (scaled : Bool) (cfg : FP.Config) (u : Nat → Word → Int)
    (a b c scale : TensorBinding.Tensor) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (read : Read) (s : State Word) (bound : Bound read c s) :
    call scaled cfg u a b c scale p (s,read)=
      (CheckedHost.call scaled cfg u a b c scale p s).map
        (fun result => (result.1,(result.2,render read c result.2))) := by
  rw [call,run_render scaled a b c scale p _ h read s bound]
  congr 1
  cases scaled <;> rfl

theorem nativeCall_render (native : CheckedNative.Instructions)
    (a b c scale : TensorBinding.Tensor) (p : Contract.Config)
    (h : InputContract Bool.true a.metadata b.metadata c.metadata scale.metadata p)
    (read : Read) (s : State Word) (bound : Bound read c s) :
    nativeCall native a b c scale p (s,read)=
      (CheckedNative.typedCall native a b c scale p s).map
        (fun result => (result.1,(result.2,render read c result.2))) :=
  run_render Bool.true a b c scale p _ h read s bound

/-- Actual returned INT32 C: exact word/readback, each of its four physical
bytes, and every byte exterior to its view. Offset views use storageOffset
through TensorBinding.pointer_index. -/
structure Post (read final : Read) (a b c : TensorBinding.Tensor) (value : Index a b → Int) : Prop where
  metadata : CheckedHost.Metadata a b c
  words : ∀ i,read32 final (c.metadata.dataPtr.toNat+4*InputAdapter.address a b i)=BitVec.ofInt 32 (value i)
  logical : ∀ i,(read32 final (physicalAddress .int32 c (InputAdapter.address a b i))).toInt=value i
  bytes : ∀ i j,j<4 → final (physicalAddress .int32 c (InputAdapter.address a b i)+j)=byte (BitVec.ofInt 32 (value i)) j
  frame : ∀ ptr,ptr<c.metadata.dataPtr.toNat ∨ c.metadata.dataPtr.toNat+4*c.metadata.numel≤ptr → final ptr=read ptr

theorem post (scaled : Bool) (a b c scale : TensorBinding.Tensor) (lc : Laws .int32 c)
    (p : Contract.Config) (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (read : Read) (initial final : State Word) (value : Index a b → Int)
    (logical : CheckedHost.OutputPost a b c initial final value)
    (range : ∀ i,(BitVec.ofInt 32 (value i)).toInt=value i) :
    Post read (render read c final) a b c value := by
  have ib : ∀ i : Index a b,InputAdapter.address a b i<c.metadata.numel := by
    intro i
    rw [←size_eq scaled a b c scale p h]
    exact linear_bound _ _ _ _ i.1.isLt i.2.isLt
  have ptr : ∀ i : Index a b,c.metadata.dataPtr.toNat+4*InputAdapter.address a b i=
      physicalAddress .int32 c (InputAdapter.address a b i) := fun i =>
    pointer_index .int32 c lc logical.1.2.2.2.2.1 _ (ib i)
  have words : ∀ i : Index a b,read32 (render read c final)
      (c.metadata.dataPtr.toNat+4*InputAdapter.address a b i)=BitVec.ofInt 32 (value i) := by
    intro i
    rw [render_read read c final _ (ib i),logical.2.1 i,Cell.read_same]
  refine ⟨logical.1,words,?_,?_,render_frame read c final⟩
  · intro i
    rw [←ptr i,words i,range i]
  · intro i j hj
    rw [←ptr i,render_at read c final _ j (ib i) hj,logical.2.1 i,Cell.read_same]

/-- Output byte addresses are aligned, representable without UINT64 wrapping,
and inside C's actual storage window, derived from the Python guards. -/
theorem output_addresses (scaled : Bool) (a b c scale : TensorBinding.Tensor) (lc : Laws .int32 c)
    (p : Contract.Config) (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (i : Index a b) (j : Nat) (hj : j<4) :
    physicalAddress .int32 c (InputAdapter.address a b i)+j<2^64 ∧
      (c.metadata.dataPtr.toNat+4*InputAdapter.address a b i)%4=0 := by
  have nonempty : ¬Empty a.metadata b.metadata := by
    have ir := i.1.isLt; have ic := i.2.isLt
    unfold Contract.Empty; omega
  have hb : InputAdapter.address a b i<c.metadata.numel := by
    rw [←size_eq scaled a b c scale p h]
    exact linear_bound _ _ _ _ i.1.isLt i.2.isLt
  have windows := (execution_of_nonempty scaled a.metadata b.metadata c.metadata scale.metadata p h nonempty).2.1
  have cp := pointer_index .int32 c lc (h.contiguous _ (member_c scaled _ _ _ _)) _ hb
  change c.metadata.dataPtr.toNat+4*InputAdapter.address a b i=physicalAddress .int32 c (InputAdapter.address a b i) at cp
  rcases windows.2.2.1 with empty | ⟨alignment,limit⟩
  · omega
  · rw [←cp]
    constructor <;> omega

theorem separated_frame (read final : Read) (input c : TensorBinding.Tensor) (width : Nat)
    (separate : Separate input.metadata width c.metadata)
    (frame : ∀ ptr,ptr<c.metadata.dataPtr.toNat ∨ c.metadata.dataPtr.toNat+4*c.metadata.numel≤ptr → final ptr=read ptr)
    (i j : Nat) (hi : i < input.metadata.numel) (hj : j<width) :
    final (input.metadata.dataPtr.toNat+width*i+j)=read (input.metadata.dataPtr.toNat+width*i+j) := by
  apply frame
  have bound := Nat.mul_le_mul_left width (show i + 1 ≤ input.metadata.numel by omega)
  rw [Nat.mul_add,Nat.mul_one] at bound
  rcases separate with empty | empty | left | right <;> omega

/-- The same physical input bytes supply every launch, including read-only
aliases. C's checked disjointness derives input preservation from the byte
frame; it is not inserted into the ordinary Tensor storage laws. -/
theorem input_frame (scaled : Bool) (a b c scale : TensorBinding.Tensor) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (read final : Read) (value : Index a b → Int) (post : Post read final a b c value) :
    (∀ i,i<a.metadata.numel → final (a.metadata.dataPtr.toNat+i)=read (a.metadata.dataPtr.toNat+i)) ∧
    (∀ i,i<b.metadata.numel → final (b.metadata.dataPtr.toNat+i)=read (b.metadata.dataPtr.toNat+i)) ∧
    (scaled=Bool.true → ∀ i j,i<scale.metadata.numel → j<4 →
      final (scale.metadata.dataPtr.toNat+4*i+j)=read (scale.metadata.dataPtr.toNat+4*i+j)) := by
  by_cases empty : Contract.Empty a.metadata b.metadata
  · have zero : c.metadata.numel=0 := by
      rw [←size_eq scaled a b c scale p h]
      rcases empty with e | e <;> simp [e]
    have frame : ∀ ptr,final ptr=read ptr := fun ptr => post.frame ptr (by rw [zero]; omega)
    exact ⟨fun _ _ => frame _,fun _ _ => frame _,fun _ _ _ _ _ => frame _⟩
  · have separate := (execution_of_nonempty scaled a.metadata b.metadata c.metadata scale.metadata p h empty).2.2
    refine ⟨?_,?_,?_⟩
    · intro i hi
      simpa only [Nat.one_mul,Nat.add_zero] using separated_frame read final a c 1 separate.1 post.frame i 0 hi (by decide)
    · intro i hi
      simpa only [Nat.one_mul,Nat.add_zero] using separated_frame read final b c 1 separate.2.1 post.frame i 0 hi (by decide)
    · intro hs i j hi hj
      exact separated_frame read final scale c 4 (separate.2.2 hs) post.frame i j hi hj

#axiomsClean input_frame
#axiomsClean run_render
#axiomsClean call_render
#axiomsClean nativeCall_render
#axiomsClean post
#axiomsClean output_addresses
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ByteHost
