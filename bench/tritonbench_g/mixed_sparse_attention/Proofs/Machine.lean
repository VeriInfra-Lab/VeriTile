import bench.tritonbench_g.mixed_sparse_attention.Proofs.Access
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Machine
open VeriTile Triton FP.Structural Source Whole
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 4000000

/-- Conservative envelope for formed tensor pointers, including masked query
and key tails, metadata lanes, loop increments, dimensions and strides. -/
def denseEnvelope (c : Config) : Nat :=
  (c.Z*c.H*c.T+(c.T+c.BM+c.BN))*c.BD+c.BD+1
def metadataEnvelope (c : Config) : Nat :=
  (c.Z*c.H*c.NR)*(c.NS+c.NV+1)+(c.NS+c.NV+c.BN)+1
def scalarEnvelope (c : Config) : Nat :=
  c.Z+c.H+c.T+c.NR+c.NS+c.NV+c.BM+c.BN+c.BD+1
def envelope (c : Config) : Nat := denseEnvelope c+metadataEnvelope c+scalarEnvelope c

def bindings (c : Config) : List (RegionName×Nat) :=
  [(c.q,4),(c.k,4),(c.v,4),(c.out,4),(Region.cast c.seq,4),
    (Region.cast c.blocks,4),(Region.cast c.offsets,4),(Region.cast c.counts,4),(Region.cast c.columns,4)]

/-- Runtime address and integer-domain contract; its bounds mention only
shapes, tile sizes, metadata capacities and actual allocation base addresses. -/
structure Domain (c : Config) (base : RegionName → Nat) : Prop where
  signed : envelope c<2147483648
  addresses : ∀ r width,(r,width)∈bindings c →
    base r%width=0 ∧ base r+width*envelope c<18446744073709551616

theorem repr32 (n : Nat) (h : n<2147483648) :
    (BitVec.ofInt 32 (n : Int)).toInt=(n : Int) := by
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega
theorem repr64 (n : Nat) (h : n<9223372036854775808) :
    (BitVec.ofInt 64 (n : Int)).toInt=(n : Int) := by
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega

structure Exact (n : Nat) : Prop where
  bound : n<2147483648
  int32 : (BitVec.ofInt 32 (n : Int)).toInt=(n : Int)
  int64 : (BitVec.ofInt 64 (n : Int)).toInt=(n : Int)
  subexpressions : ∀ x≤n,(BitVec.ofInt 32 (x : Int)).toInt=(x : Int) ∧
    (BitVec.ofInt 64 (x : Int)).toInt=(x : Int)

theorem exact_of_bound (n : Nat) (h : n<2147483648) : Exact n := by
  refine ⟨h,repr32 n h,repr64 n (by omega),?_⟩
  intro x hx
  exact ⟨repr32 x (by omega),repr64 x (by omega)⟩

theorem dense_bound (c : Config) (hz n d : Nat) (hh : hz≤c.Z*c.H)
    (hn : n≤c.T+c.BM+c.BN) (hd : d≤c.BD) : logicalAddress c hz n d<envelope c := by
  have hh':=Nat.mul_le_mul_right c.T hh
  have hm:=Nat.mul_le_mul_right c.BD (Nat.add_le_add hh' hn)
  unfold logicalAddress envelope denseEnvelope metadataEnvelope scalarEnvelope
  omega

theorem metadata_bound (c : Config) (row stride lane : Nat) (hr : row≤c.Z*c.H*c.NR)
    (hs : stride≤c.NS+c.NV+1) (hl : lane≤c.NS+c.NV+c.BN) : row*stride+lane<envelope c := by
  have hm:=Nat.mul_le_mul hr hs
  unfold envelope denseEnvelope metadataEnvelope scalarEnvelope
  omega

theorem coordinates (c : Config) (shape : Shape c) (s : State α) (metadata : Metadata c s)
    (p : Program) (hp : p∈programs (grid c)) (i : Fin c.BM)
    (ph : Bool) (n : Nat) (j : Fin c.BN) :
    p.1*c.BM+i.val<c.T+c.BM ∧
      columnValues c (start (grid c) p s) ph n j<c.T+c.BN := by
  have hbm : 0<c.BM := by have h:=shape.bm;simp only [List.mem_cons,List.not_mem_nil,or_false] at h;omega
  have hbn : 0<c.BN := by have h:=j.isLt;omega
  have hx:=((mem_programs (grid c) p.1 p.2.1 p.2.2).mp hp).1
  refine ⟨query_lane_bound c.T c.BM p.1 i.val hbm hx i.isLt,?_⟩
  obtain ⟨hr,hh,hb⟩:=Access.launched_coordinates c shape p hp
  cases ph
  · by_cases hn : n<blockCount c (start (grid c) p s)
    · have hm:=metadata.blockOffset p.2.1 hh p.1 hr n hn
      have hs:=metadata.sequence _ hb
      change (if n<blockCount c (start (grid c) p s) then _ else 0)+j.val<_
      rw [if_pos hn]
      exact Nat.add_lt_add (lt_of_lt_of_le hm hs) j.isLt
    · change (if n<blockCount c (start (grid c) p s) then _ else 0)+j.val<_
      rw [if_neg hn,zero_add]
      exact lt_of_lt_of_le j.isLt (by omega)
  · by_cases ha : keyActive c (start (grid c) p s) Bool.true n j
    · exact lt_of_lt_of_le (Access.key_bound c shape s metadata p hp Bool.true n j ha) (by omega)
    · change ¬(n+j.val<columnCount c (start (grid c) p s) ∧ n<columnCount c (start (grid c) p s)) at ha
      change (if n+j.val<columnCount c (start (grid c) p s) ∧
        n<columnCount c (start (grid c) p s) then _ else 0)<_
      rw [if_neg ha]
      omega

structure Closed (c : Config) (base : RegionName → Nat) : Prop where
  dense : ∀ hz≤c.Z*c.H,∀ n≤c.T+c.BM+c.BN,∀ d≤c.BD,Exact (logicalAddress c hz n d)
  metadata : ∀ row≤c.Z*c.H*c.NR,∀ stride≤c.NS+c.NV+1,∀ lane≤c.NS+c.NV+c.BN,
    Exact (row*stride+lane)
  scalars : ∀ x≤c.Z+c.H+c.T+c.NR+c.NS+c.NV+c.BM+c.BN+c.BD,Exact x
  physical : ∀ r width,(r,width)∈bindings c →∀ off<envelope c,
    width*off<9223372036854775808 ∧ base r+width*off<18446744073709551616 ∧
      (BitVec.ofInt 64 ((width*off : Nat) : Int)).toInt=((width*off : Nat) : Int)

theorem closed (c : Config) (base : RegionName → Nat) (dom : Domain c base) : Closed c base := by
  refine ⟨?_,?_,?_,?_⟩
  · intro hz hh n hn d hd
    exact exact_of_bound _ (lt_trans (dense_bound c hz n d hh hn hd) dom.signed)
  · intro row hr stride hs lane hl
    exact exact_of_bound _ (lt_trans (metadata_bound c row stride lane hr hs hl) dom.signed)
  · intro x hx
    apply exact_of_bound
    have h:=dom.signed
    unfold envelope scalarEnvelope at h
    omega
  · intro r width hw off ho
    have hwidth : width≤4 := by
      simp only [bindings,List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at hw
      rcases hw with h|h|h|h|h|h|h|h|h <;> omega
    have hm:=Nat.mul_le_mul_right off hwidth
    have hs:=dom.signed
    have hb:=(dom.addresses r width hw).2
    have hmon:=Nat.mul_le_mul_left width (Nat.le_of_lt ho)
    have hdisp : width*off<9223372036854775808 := by omega
    exact ⟨hdisp,by omega,repr64 _ hdisp⟩

#axiomsClean coordinates
#axiomsClean closed
#specNonCircular Domain avoiding [FP.Structural.exec,Whole.invoke,Source.finalValues]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Machine
