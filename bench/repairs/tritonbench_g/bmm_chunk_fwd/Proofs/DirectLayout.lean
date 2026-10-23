import bench.repairs.tritonbench_g.bmm_chunk_fwd.Target
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.Contract
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.HostReturned
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.Finite

namespace VeriTile.Bench.Repairs.BmmChunkFwd.Direct
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
open Classical
set_option maxHeartbeats 3000000

def inputIndex (h : Host.Shape) (b r g k : Nat) := ((b*h.seqlen+r)*h.groups+g)*h.K+k

theorem input_offset (h : Host.Shape) (b r g k : Nat) (hg : g<h.groups) :
    Host.inputOffset (Host.prepared h (Host.contiguous h)) b r g k=inputIndex h b r g k := by
  rw [contiguous_prepared]
  cases grouped : h.grouped
  · have g0 : g=0 := by simpa [Host.Shape.groups,grouped] using hg
    subst g
    simp only [Host.inputOffset,Host.contiguous,inputIndex,Host.Shape.groups,grouped,
      Bool.false_eq_true,↓reduceIte,Nat.mul_one,Nat.mul_zero,Nat.zero_mul,Nat.add_zero]
    ring
  · simp only [Host.inputOffset,Host.contiguous,inputIndex,Host.Shape.groups,grouped,↓reduceIte,Nat.mul_one]
    ring

theorem input_bound (p : Metadata) (h : InputContract p) (b r g k : Nat)
    (hb : b<(shape p).batch) (hr : r<(shape p).seqlen) (hg : g<(shape p).groups) (hk : k<(shape p).K) :
    inputIndex (shape p) b r g k<(p.tensor .a).numel := by
  rw [input_size p h]
  exact Host.mul_add_bound (Host.mul_add_bound (Host.mul_add_bound hb hr) hg) hk

theorem a_stride (p : Metadata) (h : InputContract p) (b r g k : Nat)
    (hb : b<(shape p).batch) (hr : r<(shape p).seqlen) (hg : g<(shape p).groups) (hk : k<(shape p).K) :
    p.strides .a=Host.contiguous (shape p) :=
  h.strides.1.2 (Nat.zero_lt_of_lt (input_bound p h b r g k hb hr hg hk))

theorem b_stride (p : Metadata) (h : InputContract p) (b r g k : Nat)
    (hb : b<(shape p).batch) (hr : r<(shape p).seqlen) (hg : g<(shape p).groups) (hk : k<(shape p).K) :
    p.strides .b=Host.contiguous (shape p) := by
  have same : (p.tensor .b).numel=(p.tensor .a).numel := by simp [VeriTile.Torch.TensorMetadata.numel,h.shapes.2]
  apply h.strides.2.1.2
  rw [same]
  exact Nat.zero_lt_of_lt (input_bound p h b r g k hb hr hg hk)

theorem seq_stride (p : Metadata) (h : InputContract p) (seq : p.hasSeq=Bool.true)
    (b r : Nat) (hb : b<(shape p).batch) (hr : r<(shape p).seqlen) :
    (p.strides .seq).batch=(shape p).seqlen ∧ (p.strides .seq).row=1 := by
  apply h.strides.2.2 seq
  have bound := Host.mul_add_bound hb hr
  have size : (p.tensor .seq).numel=(shape p).batch*(shape p).seqlen := by
    unfold VeriTile.Torch.TensorMetadata.numel
    rw [seq_shape p h seq]; simp
  rw [size]
  exact Nat.zero_lt_of_lt bound

/-- Ordinary fresh output allocation and representable Tensor data spans.
These fields describe actual storage before execution, without any kernel safety
or value premise. Empty absent metadata has no occupied span. -/
structure Allocation (p : Metadata) where
  pointer : Nat
  object : Nat
  inputObjects : Role → Nat
  fresh : ∀ r,Used p r → object≠inputObjects r
  inputsFit : ∀ r,Used p r → base p r+bytes r*(p.tensor r).numel≤2^64
  outputFit : pointer+4*(shape p).outputSize≤2^64
  separate : ∀ r,Used p r →
    pointer+4*(shape p).outputSize≤base p r ∨ base p r+bytes r*(p.tensor r).numel≤pointer

def extent (p : Metadata) (r : RegionName) : Nat :=
  if r="A" then (p.tensor .a).numel else if r="B" then (p.tensor .b).numel
  else if r="Seq" then if p.hasSeq then (p.tensor .seq).numel else 0
  else if r="Out" then (shape p).outputSize else 0

def flat (p : Metadata) (a : Allocation p) : FlatAlloc where
  flat := "heap"
  regions := ["A","B","Seq","Out"]
  base := fun r => if r="A" then base p .a else if r="B" then base p .b
    else if r="Seq" then (if p.hasSeq then base p .seq else 0) else if r="Out" then a.pointer else 0
  extent := extent p

theorem extent_closed (p : Metadata) (r : RegionName)
    (outside : r∉(["A","B","Seq","Out"] : List RegionName)) : extent p r=0 := by
  simp only [List.mem_cons,List.not_mem_nil,or_false,not_or] at outside
  simp [extent,outside.1,outside.2.1,outside.2.2.1,outside.2.2.2]

theorem flat_disjoint (p : Metadata) (h : InputContract p) (a : Allocation p) : (flat p a).Disjoint := by
  have usedA : Used p .a := Or.inl (by decide)
  have usedB : Used p .b := Or.inl (by decide)
  have ab := h.separate .a .b usedA usedB (by decide)
  have ao := a.separate .a usedA
  have bo := a.separate .b usedB
  intro r hr s hs different
  simp only [flat,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl|rfl|rfl|rfl <;> rcases hs with rfl|rfl|rfl|rfl
  all_goals try { exact False.elim (different rfl) }
  all_goals simp only [flat,extent,bytes,ite_true,show ("A":RegionName)≠"B" by decide,
    show ("A":RegionName)≠"Seq" by decide,show ("A":RegionName)≠"Out" by decide,
    show ("B":RegionName)≠"A" by decide,show ("B":RegionName)≠"Seq" by decide,
    show ("B":RegionName)≠"Out" by decide,show ("Seq":RegionName)≠"A" by decide,
    show ("Seq":RegionName)≠"B" by decide,show ("Seq":RegionName)≠"Out" by decide,
    show ("Out":RegionName)≠"A" by decide,show ("Out":RegionName)≠"B" by decide,
    show ("Out":RegionName)≠"Seq" by decide,ite_false]
  all_goals try { simp only [bytes] at ab ao bo; omega }
  all_goals cases seq : p.hasSeq
  all_goals simp only [seq,Bool.false_eq_true,↓reduceIte,Nat.add_zero]
  all_goals try omega
  all_goals
    have usedS : Used p .seq := Or.inr seq
    have as' := h.separate .a .seq usedA usedS (by decide)
    have bs := h.separate .b .seq usedB usedS (by decide)
    have so := a.separate .seq usedS
    simp only [bytes] at as' bs so
    omega

theorem finite_sizes (p : Metadata) (h : InputContract p) (nonempty : 0<(shape p).outputSize) :
    Finite.AllocatedSizes (shape p) := by
  have hi := h.extents.1 .a (Or.inl (by decide))
  rw [input_size p h] at hi
  simp only [bytes] at hi
  exact ⟨nonempty,by have := h.extents.2; omega,by omega⟩

abbrev RealData := Mathematical.Inputs

def realState (p : Metadata) (d : RealData) : BlockState where
  mem := fun r off => if r="A" then .real (d.a off) else if r="B" then .real (d.b off)
    else if r="Seq" then MemCell.of .int (d.seq off).toInt else .real (d.output off)
  regs := fun _ _ _ => none
  pids := fun _ => 0

def realInputs (p : Metadata) (d : RealData) : Host.Inputs (call p) where
  a := fun b r g k => d.a (inputIndex (shape p) b r g k)
  b := fun b r g k => d.b (inputIndex (shape p) b r g k)
  seq := fun b r => (d.seq (b*(shape p).seqlen+r)).toInt

theorem real_binding (p : Metadata) (h : InputContract p) (d : RealData)
    (nonempty : 0<(shape p).outputSize) :
    Host.TensorInputs (call p) (realInputs p d) (realState p d) (extent p) := by
  have hg := (Finite.positive_dimensions _ (finite_sizes p h nonempty)).2.2.1
  refine ⟨hg,by change ("A":RegionName)≠"Out"; decide,by change ("B":RegionName)≠"Out"; decide,by change ("Seq":RegionName)≠"Out"; decide,?_,?_,?_,?_,?_,?_⟩
  · intro b r g k hb hr hg
    change Host.inputOffset (Host.prepared (shape p) (p.strides .a)) b r g k<_
    rw [a_stride p h b r g k hb hr hg k.isLt,input_offset (shape p) b r g k hg]
    change inputIndex (shape p) b r g k<(p.tensor .a).numel
    exact input_bound p h b r g k hb hr hg k.isLt
  · intro b r g k hb hr hg
    change Host.inputOffset (Host.prepared (shape p) (p.strides .b)) b r g k<_
    rw [b_stride p h b r g k hb hr hg k.isLt,input_offset (shape p) b r g k hg]
    have same : (p.tensor .b).numel=(p.tensor .a).numel := by simp [VeriTile.Torch.TensorMetadata.numel,h.shapes.2]
    change inputIndex (shape p) b r g k<(p.tensor .b).numel
    rw [same]
    exact input_bound p h b r g k hb hr hg k.isLt
  · intro seq b r hb hr
    have size : (p.tensor .seq).numel=(shape p).batch*(shape p).seqlen := by
      unfold VeriTile.Torch.TensorMetadata.numel
      rw [seq_shape p h seq]; simp
    change b*(p.strides .seq).batch+r*(p.strides .seq).row<(if p.hasSeq then (p.tensor .seq).numel else 0)
    have hs : p.hasSeq=Bool.true := seq
    rw [(seq_stride p h hs b r hb hr).1,(seq_stride p h hs b r hb hr).2]
    simpa [hs,size] using Host.mul_add_bound hb hr
  · intro b r g k hb hr hg
    change (realState p d).readMem "A" (Host.inputOffset (Host.prepared (shape p) (p.strides .a)) b r g k)=_
    rw [a_stride p h b r g k hb hr hg k.isLt,input_offset (shape p) b r g k hg]
    rfl
  · intro b r g k hb hr hg
    change (realState p d).readMem "B" (Host.inputOffset (Host.prepared (shape p) (p.strides .b)) b r g k)=_
    rw [b_stride p h b r g k hb hr hg k.isLt,input_offset (shape p) b r g k hg]
    rfl
  · intro seq b r hb hr
    change (realState p d).readMemValue .int "Seq" (b*(p.strides .seq).batch+r*(p.strides .seq).row)=(d.seq (b*(shape p).seqlen+r)).toInt
    rw [(seq_stride p h seq b r hb hr).1,(seq_stride p h seq b r hb hr).2]
    simp only [Nat.mul_one]
    rfl

#axiomsClean real_binding
#axiomsClean flat_disjoint
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.Direct
