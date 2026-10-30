import bench.repairs.tritonbench_g.attention_kernel.Proofs.Host
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Machine
open VeriTile Triton Source Whole
noncomputable section
set_option maxHeartbeats 4000000

def offset (s : Strides) (b h n d : Nat) : Nat :=
  b*s.batch+h*s.head+n*s.row+d*s.dim

/-- Actual data pointers. Both supported storage formats occupy two bytes. -/
structure Bases where
  (q k v bias out : Nat)

def bases (b : Bases) : List Nat := [b.q,b.k,b.v,b.bias,b.out]

/-- Launch, dense-row geometry and physical-address validity are ordinary
contracts, required only after the actual empty-input branch. -/
structure Domain (p : Parameters) (b : Bases) : Prop where
  launch : Live p → Guard p
  addresses : Live p → ∀ base∈bases b,
    base%2=0 ∧ base+2*envelope p<18446744073709551616

theorem repr32 (n : Nat) (h : n<2147483648) :
    (BitVec.ofInt 32 (n : Int)).toInt=(n : Int) := by
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega

theorem repr64 (n : Nat) (h : n<9223372036854775808) :
    (BitVec.ofInt 64 (n : Int)).toInt=(n : Int) := by
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega

structure Exact (n : Nat) : Prop where
  signed32 : n<2147483648
  int32 : (BitVec.ofInt 32 (n : Int)).toInt=(n : Int)
  int64 : (BitVec.ofInt 64 (n : Int)).toInt=(n : Int)
  subexpressions : ∀ x≤n,(BitVec.ofInt 32 (x : Int)).toInt=(x : Int) ∧
    (BitVec.ofInt 64 (x : Int)).toInt=(x : Int)

theorem exact_of_bound (n : Nat) (h : n<2147483648) : Exact n := by
  refine ⟨h,repr32 n h,repr64 n (by omega),?_⟩
  intro x hx
  exact ⟨repr32 x (by omega),repr64 x (by omega)⟩

/-- Covers active and inactive formed offsets, bias columns, pointer advances,
and all their smaller nonnegative arithmetic subexpressions. -/
theorem formed_exact (p : Parameters) (g : Guard p) (s : Strides)
    (hs : s∈[p.q,p.k,p.v,p.o,p.bias]) (b h n d c : Nat)
    (hb : b≤p.batches) (hh : h≤p.heads) (hn : n≤p.length+p.bm+p.bn)
    (hd : d≤p.dim) (hc : c≤127) : Exact (offset s b h n d+c) := by
  apply exact_of_bound
  have h:=source_integer_bound p g s hs b h n d hb hh hn hd
  dsimp [offset]
  omega

theorem source_coordinates (p : Parameters) (g : Guard p) (x y : Nat)
    (hx : x<(p.length+p.bm-1)/p.bm) (hy : y<p.batches*p.heads)
    (row r k : Nat) (hr : row<p.length) (hlane : r<p.bm) (hk : k<p.bn) :
    y/p.heads<p.batches ∧ y%p.heads<p.heads ∧
    x*p.bm+r<p.length+p.bm ∧ row+k<p.length+p.bn ∧ row+p.bn≤p.length+p.bn := by
  have head:=head_decode p y hy
  have lane:=query_lane_bound p.length p.bm x r g.bm hx hlane
  exact ⟨head.2.1,head.2.2,lane,by omega,by omega⟩

theorem byte_extent (p : Parameters) (b : Bases) (dom : Domain p b) (live : Live p)
    (base : Nat) (hb : base∈bases b) (n : Nat) (hn : n<envelope p) :
    2*n<9223372036854775808 ∧ base+2*n<18446744073709551616 ∧
    (BitVec.ofInt 64 ((2*n : Nat) : Int)).toInt=((2*n : Nat) : Int) := by
  have hg:=(dom.launch live).signed
  have hdisp : 2*n<9223372036854775808 := by omega
  have physical:=(dom.addresses live base hb).2
  exact ⟨hdisp,by omega,repr64 _ hdisp⟩

structure Closed (p : Parameters) (b : Bases) : Prop where
  offsets : Live p → ∀ s∈[p.q,p.k,p.v,p.o,p.bias],∀ z≤p.batches,∀ h≤p.heads,
    ∀ n≤p.length+p.bm+p.bn,∀ d≤p.dim,∀ c≤127,Exact (offset s z h n d+c)
  physical : Live p → ∀ base∈bases b,∀ n<envelope p,
    2*n<9223372036854775808 ∧ base+2*n<18446744073709551616 ∧
    (BitVec.ofInt 64 ((2*n : Nat) : Int)).toInt=((2*n : Nat) : Int)

theorem closed (p : Parameters) (b : Bases) (dom : Domain p b) : Closed p b := by
  refine ⟨?_,?_⟩
  · intro live s hs z hz h hh n hn d hd c hc
    exact formed_exact p (dom.launch live) s hs z h n d c hz hh hn hd hc
  · intro live base hb n hn
    exact byte_extent p b dom live base hb n hn

#axiomsClean closed
#specNonCircular Domain avoiding [FP.Structural.exec,Whole.invoke]
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Machine
