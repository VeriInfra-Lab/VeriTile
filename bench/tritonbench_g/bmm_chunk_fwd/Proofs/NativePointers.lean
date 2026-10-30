import bench.tritonbench_g.bmm_chunk_fwd.Proofs.SourcePointers

namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativePointers
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open SourcePointers
noncomputable section

/-- Ordinary tensor storage metadata. `pointer` is the tensor's data pointer,
already including its storage offset; byte displacements add it exactly once. -/
structure Buffer where
  pointer : Nat
  bytes : Nat
  extent : Nat
  capacity : Nat
  width : 0<bytes
  allocated : bytes*extent≤capacity
  bounded : capacity<2^63
  address : pointer+capacity≤2^64

theorem Buffer.small (b : Buffer) : b.extent<2^63 :=
  lt_of_le_of_lt ((Nat.le_mul_of_pos_left b.extent b.width).trans b.allocated) b.bounded

/-- Local direct-buffer/allocation layout contracts, quantified over logical tensor
coordinates. These assert ordinary initial storage bounds, never the safety or
result of a kernel execution. No special shape cutoff is imposed. -/
structure Layout (h : Host.Call) where
  a : Buffer
  b : Buffer
  seq : Buffer
  out : Buffer
  aWidth : a.bytes=4
  bWidth : b.bytes=4
  seqWidth : seq.bytes=4 ∨ seq.bytes=8
  outWidth : out.bytes=1 ∨ out.bytes=2 ∨ out.bytes=4 ∨ out.bytes=8
  aBounds : ∀ b r g k, b<h.shape.batch → r<h.shape.seqlen → g<h.shape.groups → k<h.shape.K →
    FPGrid.inputOffset (Host.prepared h.shape h.aStrides) b r g k<a.extent
  bBounds : ∀ batch r g k, batch<h.shape.batch → r<h.shape.seqlen → g<h.shape.groups → k<h.shape.K →
    FPGrid.inputOffset (Host.prepared h.shape h.bStrides) batch r g k<b.extent
  seqBounds : h.hasSeq=Bool.true → ∀ b r, b<h.shape.batch → r<h.shape.seqlen →
    b*h.seqBatch+r*h.seqRow<seq.extent
  outExtent : out.extent=h.shape.outputSize

def base (h : Host.Call) (v : Host.Strides) (d : Coordinates) : BitVec 64 :=
  BitVec.ofNat 64 d.batch*BitVec.ofNat 64 v.batch+
  BitVec.ofNat 64 d.chunk*BitVec.ofNat 64 h.shape.chunk*BitVec.ofNat 64 v.row+
  BitVec.ofNat 64 d.head*BitVec.ofNat 64 v.head

def aInitial (h : Host.Call) (c : Host.Config) (d : Coordinates) (i e : Nat) : BitVec 64 :=
  base h (Host.prepared h.shape h.aStrides) d+
    ((BitVec.ofNat 64 d.m*BitVec.ofNat 64 c.M+BitVec.ofNat 64 i)*
      BitVec.ofNat 64 (Host.prepared h.shape h.aStrides).row+
      BitVec.ofNat 64 e*BitVec.ofNat 64 (Host.prepared h.shape h.aStrides).column)

def bInitial (h : Host.Call) (c : Host.Config) (d : Coordinates) (e j : Nat) : BitVec 64 :=
  base h (Host.prepared h.shape h.bStrides) d+
    (BitVec.ofNat 64 e*BitVec.ofNat 64 (Host.prepared h.shape h.bStrides).column+
      (BitVec.ofNat 64 d.n*BitVec.ofNat 64 c.N+BitVec.ofNat 64 j)*
      BitVec.ofNat 64 (Host.prepared h.shape h.bStrides).row)

/-- Actual recurrent pointer advance; includes the last advance after the last
load, whose masked or terminal value need not be within the allocation. -/
def advance (initial : BitVec 64) (block stride : Nat) : Nat → BitVec 64
  | 0 => initial
  | q+1 => advance initial block stride q+BitVec.ofNat 64 block*BitVec.ofNat 64 stride

theorem advance_formula (initial : BitVec 64) (block stride q : Nat) :
    advance initial block stride q=initial+BitVec.ofNat 64 (q*block*stride) := by
  induction q with
  | zero => simp [advance]
  | succ q ih =>
    rw [advance,ih]
    simp only [Nat.add_mul,Nat.one_mul,BitVec.ofNat_add,BitVec.ofNat_mul,BitVec.add_assoc]

def aWord (h : Host.Call) (c : Host.Config) (d : Coordinates) (q i e : Nat) :=
  advance (aInitial h c d i e) c.K (Host.prepared h.shape h.aStrides).column q

def bWord (h : Host.Call) (c : Host.Config) (d : Coordinates) (q e j : Nat) :=
  advance (bInitial h c d e j) c.K (Host.prepared h.shape h.bStrides).column q

theorem a_formula (h : Host.Call) (c : Host.Config) (d : Coordinates) (q i e : Nat) :
    aWord h c d q i e=BitVec.ofNat 64 (aOffset h c d q i e) := by
  simp only [aWord,advance_formula,aInitial,base,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  congr 1
  simp only [aOffset,FPGrid.inputOffset,row]
  ring

theorem b_formula (h : Host.Call) (c : Host.Config) (d : Coordinates) (q e j : Nat) :
    bWord h c d q e j=BitVec.ofNat 64 (bOffset h c d q e j) := by
  simp only [bWord,advance_formula,bInitial,base,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  congr 1
  simp only [bOffset,FPGrid.inputOffset,col]
  ring

def seqWord (h : Host.Call) (d : Coordinates) (lane : Nat) : BitVec 64 :=
  (BitVec.ofNat 64 d.batch*BitVec.ofNat 64 h.seqBatch+
    BitVec.ofNat 64 d.chunk*BitVec.ofNat 64 h.shape.chunk*BitVec.ofNat 64 h.seqRow)+
    BitVec.ofNat 64 lane*BitVec.ofNat 64 h.seqRow

theorem seq_formula (h : Host.Call) (d : Coordinates) (lane : Nat) :
    seqWord h d lane=BitVec.ofNat 64 (seqOffset h d lane) := by
  simp only [seqWord,seqOffset,seqBase,←BitVec.ofNat_mul,←BitVec.ofNat_add]

def outWord (h : Host.Call) (c : Host.Config) (d : Coordinates) (i j : Nat) : BitVec 64 :=
  (BitVec.ofNat 64 d.batch*BitVec.ofNat 64 ((h.shape.chunks*h.shape.groups)*h.shape.chunk*h.shape.chunk)+
    BitVec.ofNat 64 d.chunk*BitVec.ofNat 64 (h.shape.groups*h.shape.chunk*h.shape.chunk)+
    BitVec.ofNat 64 d.head*BitVec.ofNat 64 (if h.shape.grouped then h.shape.chunk*h.shape.chunk else 0))+
    (BitVec.ofNat 64 h.shape.chunk*(BitVec.ofNat 64 d.m*BitVec.ofNat 64 c.M+BitVec.ofNat 64 i)+
      (BitVec.ofNat 64 d.n*BitVec.ofNat 64 c.N+BitVec.ofNat 64 j)*BitVec.ofNat 64 1)

theorem out_formula (h : Host.Call) (c : Host.Config) (d : Coordinates) (i j : Nat) :
    outWord h c d i j=BitVec.ofNat 64 (outOffset h c d i j) := by
  simp only [outWord,outOffset,row,col,←BitVec.ofNat_mul,←BitVec.ofNat_add]

def Address (b : Buffer) (word : BitVec 64) (off : Nat) : Prop :=
  word.toNat=off ∧
  (BitVec.ofNat 64 b.pointer+BitVec.ofNat 64 b.bytes*word).toNat=b.pointer+b.bytes*off ∧
  b.pointer+b.bytes*off+b.bytes≤b.pointer+b.capacity

theorem address (b : Buffer) (word : BitVec 64) (off : Nat)
    (hw : word=BitVec.ofNat 64 off) (inside : off<b.extent) : Address b word off := by
  have bound := b.small
  have exactWord : word.toNat=off := by
    rw [hw,BitVec.toNat_ofNat]
    apply Nat.mod_eq_of_lt
    norm_num at bound ⊢
    omega
  refine ⟨exactWord,?_⟩
  rw [hw]
  exact Finite.active_address b.pointer b.bytes off b.capacity b.width
    ((Nat.mul_le_mul_left b.bytes (Nat.succ_le_of_lt inside)).trans b.allocated) b.address

theorem input_address (buffer : Buffer) (b r g k : Nat) (v : Host.Strides)
    (word : BitVec 64) (hw : word=BitVec.ofNat 64 (FPGrid.inputOffset v b r g k))
    (inside : FPGrid.inputOffset v b r g k<buffer.extent) :
    Address buffer word (FPGrid.inputOffset v b r g k) := by
  have finite : word=Finite.offset b r g k v := by rw [hw,Finite.offset_formula]; rfl
  refine ⟨?_,?_⟩
  · rw [finite]
    exact Finite.active_offset b r g k v buffer.extent inside buffer.small
  · rw [hw]
    exact Finite.active_address buffer.pointer buffer.bytes _ buffer.capacity buffer.width
      ((Nat.mul_le_mul_left buffer.bytes (Nat.succ_le_of_lt inside)).trans buffer.allocated) buffer.address

theorem coordinate_bounds (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (sizes : Finite.AllocatedSizes h.shape) (p : Geometry.PID h c) :
    (coordinates h c p).batch<h.shape.batch ∧
    (coordinates h c p).head<h.shape.groups ∧
    ∀ lane, lane<limit h (coordinates h c p) →
      (coordinates h c p).chunk*h.shape.chunk+lane<h.shape.seqlen := by
  have hg := (Finite.positive_dimensions h.shape sizes).2.2.1
  have mod := Nat.mod_lt (Geometry.p2 p) hg
  have rem : Geometry.p2 p-Geometry.p2 p/h.shape.groups*h.shape.groups=Geometry.p2 p%h.shape.groups := by
    have he := Nat.mod_add_div (Geometry.p2 p) h.shape.groups
    rw [Nat.mul_comm] at he
    omega

  refine ⟨(p ⟨1,by change 1<3; decide⟩).isLt,?_,?_⟩
  · simpa only [coordinates,rem] using mod
  · intro lane hl
    have base := (Finite.source_indices h c hc sizes).chunkBase p
    have lim := Nat.min_le_right h.shape.chunk (h.shape.seqlen-Geometry.p2 p/h.shape.groups*h.shape.chunk)
    change lane<min h.shape.chunk (h.shape.seqlen-Geometry.p2 p/h.shape.groups*h.shape.chunk) at hl
    change Geometry.p2 p/h.shape.groups*h.shape.chunk+lane<h.shape.seqlen
    omega

theorem a_active (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (sizes : Finite.AllocatedSizes h.shape) (layout : Layout h) (p : Geometry.PID h c)
    (q i e : Nat) (hr : row c (coordinates h c p) i<limit h (coordinates h c p))
    (hk : q*c.K+e<h.shape.K) :
    Address layout.a (aWord h c (coordinates h c p) q i e) (aOffset h c (coordinates h c p) q i e) := by
  obtain ⟨hb,hg,rows⟩ := coordinate_bounds h c hc sizes p
  apply input_address layout.a _ _ _ _ _ _ (a_formula h c _ q i e)
  exact layout.aBounds _ _ _ _ hb (rows _ hr) hg hk

theorem b_active (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (sizes : Finite.AllocatedSizes h.shape) (layout : Layout h) (p : Geometry.PID h c)
    (q e j : Nat) (hr : col c (coordinates h c p) j<limit h (coordinates h c p))
    (hk : q*c.K+e<h.shape.K) :
    Address layout.b (bWord h c (coordinates h c p) q e j) (bOffset h c (coordinates h c p) q e j) := by
  obtain ⟨hb,hg,rows⟩ := coordinate_bounds h c hc sizes p
  apply input_address layout.b _ _ _ _ _ _ (b_formula h c _ q e j)
  exact layout.bBounds _ _ _ _ hb (rows _ hr) hg hk

theorem seq_active (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (sizes : Finite.AllocatedSizes h.shape) (layout : Layout h) (p : Geometry.PID h c)
    (lane : Nat) (hseq : h.hasSeq=Bool.true) (hr : lane<limit h (coordinates h c p)) :
    Address layout.seq (seqWord h (coordinates h c p) lane) (seqOffset h (coordinates h c p) lane) := by
  obtain ⟨hb,_,rows⟩ := coordinate_bounds h c hc sizes p
  apply address _ _ _ (seq_formula h _ lane)
  have inside := layout.seqBounds hseq _ _ hb (rows _ hr)
  simpa only [seqOffset,seqBase,Nat.add_mul,Nat.add_assoc] using inside

theorem out_normal (h : Host.Call) (c : Host.Config) (p : Geometry.PID h c) (i j : Nat) :
    outOffset h c (coordinates h c p) i j=
      ((Geometry.p1 p*(h.shape.chunks*h.shape.groups)+Geometry.p2 p)*h.shape.chunk+
        row c (coordinates h c p) i)*h.shape.chunk+col c (coordinates h c p) j := by
  have hh : Geometry.p2 p/h.shape.groups*h.shape.groups+
      (Geometry.p2 p-Geometry.p2 p/h.shape.groups*h.shape.groups)=Geometry.p2 p :=
    Nat.add_sub_of_le (Nat.div_mul_le_self _ _)
  unfold outOffset
  cases hg : h.shape.grouped
  · have hgroup : h.shape.groups=1 := by simp [Host.Shape.groups,hg]
    simp only [hg,Bool.false_eq_true,↓reduceIte,coordinates,hgroup,Nat.div_one,
      Nat.mul_one,Nat.mul_zero,Nat.add_zero]
    ring
  · simp only [hg,↓reduceIte,coordinates,Nat.mul_one]
    calc
      _ = ((Geometry.p1 p*(h.shape.chunks*h.shape.groups)+
        (Geometry.p2 p/h.shape.groups*h.shape.groups+
          (Geometry.p2 p-Geometry.p2 p/h.shape.groups*h.shape.groups)))*h.shape.chunk+
        row c (coordinates h c p) i)*h.shape.chunk+col c (coordinates h c p) j := by
          simp only [row,col,coordinates]
          ring
      _ = _ := by rw [hh]; rfl

theorem mul_add_bound {a b n x : Nat} (ha : a<b) (hx : x<n) : a*n+x<b*n := by
  have hh := Nat.mul_le_mul_right n (Nat.succ_le_of_lt ha)
  nlinarith

theorem out_active (h : Host.Call) (c : Host.Config) (layout : Layout h)
    (p : Geometry.PID h c) (i j : Nat)
    (hr : row c (coordinates h c p) i<h.shape.chunk)
    (hc : col c (coordinates h c p) j<h.shape.chunk) :
    Address layout.out (outWord h c (coordinates h c p) i j) (outOffset h c (coordinates h c p) i j) := by
  apply address _ _ _ (out_formula h c _ i j)
  rw [layout.outExtent,out_normal]
  have hb : Geometry.p1 p<h.shape.batch := (p ⟨1,by change 1<3; decide⟩).isLt
  exact mul_add_bound (mul_add_bound (mul_add_bound hb (Geometry.p2_lt p)) hr) hc

#axiomsClean a_formula
#axiomsClean b_formula
#axiomsClean a_active
#axiomsClean b_active
#axiomsClean seq_active
#axiomsClean out_active
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativePointers

#guard_msgs (drop info) in
#auditModuleAxioms
