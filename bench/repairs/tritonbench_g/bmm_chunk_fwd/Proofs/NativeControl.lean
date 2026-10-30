import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.Finite
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.NativeWhole

namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeControl
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section

/-- Successful ordinary allocation and CUDA launch metadata. These are only
used after the wrapper's nonempty/setup branch; they are not Python filters. -/
structure Metadata (h : Host.Call) (c : Host.Config) : Prop where
  sizes : Finite.AllocatedSizes h.shape
  gridX : Host.cdiv h.shape.chunk c.M*Host.cdiv h.shape.chunk c.N≤2147483647
  gridY : h.shape.batch≤65535
  gridZ : h.shape.chunks*h.shape.groups≤65535

def pid (n : Nat) : BitVec 64 := BitVec.ofInt 64 (BitVec.ofNat 32 n).toInt
def ceil (n b : Nat) : BitVec 64 :=
  (BitVec.ofNat 64 n+BitVec.ofNat 64 b-1).sdiv (BitVec.ofNat 64 b)

theorem nat64 (n : Nat) (hn : n<9223372036854775808) : (BitVec.ofNat 64 n).toNat=n := by
  simp only [BitVec.toNat_ofNat]; exact Nat.mod_eq_of_lt (by omega)
theorem msb64 (n : Nat) (hn : n<9223372036854775808) : (BitVec.ofNat 64 n).msb=Bool.false := by
  rw [BitVec.msb_eq_decide,nat64 n hn]
  norm_num
  omega
theorem division (a b : Nat) (ha : a<9223372036854775808) (hb : b<9223372036854775808) :
    (BitVec.ofNat 64 a).sdiv (BitVec.ofNat 64 b)=BitVec.ofNat 64 (a/b) := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_sdiv,msb64 a ha,msb64 b hb]
  change (BitVec.ofNat 64 a).toNat / (BitVec.ofNat 64 b).toNat = _
  rw [nat64 a ha,nat64 b hb]
  exact (nat64 (a/b) (lt_of_le_of_lt (Nat.div_le_self _ _) ha)).symm
theorem remainder (a b : Nat) (ha : a<9223372036854775808) (hb : b<9223372036854775808) :
    (BitVec.ofNat 64 a).srem (BitVec.ofNat 64 b)=BitVec.ofNat 64 (a%b) := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_srem,msb64 a ha,msb64 b hb]
  simp only [nat64 a ha,nat64 b hb]
  exact (nat64 (a%b) (lt_of_le_of_lt (Nat.mod_le _ _) ha)).symm
theorem pid_exact (n : Nat) (hn : n<2147483648) : pid n=BitVec.ofNat 64 n := by
  have hnat : (BitVec.ofNat 32 n).toNat=n := by
    simp only [BitVec.toNat_ofNat]; apply Nat.mod_eq_of_lt; omega
  have hint : (BitVec.ofNat 32 n).toInt=n := by
    rw [BitVec.toInt_eq_toNat_of_lt,hnat]
    rw [hnat]; norm_num; omega
  simp only [pid,hint,BitVec.ofInt_natCast]

theorem ceil_exact (n b : Nat) (hb : 0<b) (hnb : n+b<9223372036854775808) :
    ceil n b=BitVec.ofNat 64 (Host.cdiv n b) := by
  unfold ceil
  rw [←BitVec.ofNat_add]
  change (BitVec.ofNat 64 (n+b)-BitVec.ofNat 64 1).sdiv _= _
  rw [BitVec.ofNat_sub_ofNat_of_le (n+b) 1 (by omega) (by omega)]
  exact division (n+b-1) b (by omega) (by omega)

def b (h : Host.Call) (c : Host.Config) (p : Geometry.PID h c) := pid (Geometry.p1 p)
def ch (h : Host.Call) (c : Host.Config) (p : Geometry.PID h c) :=
  (pid (Geometry.p2 p)).sdiv (BitVec.ofNat 64 h.shape.groups)
def head (h : Host.Call) (c : Host.Config) (p : Geometry.PID h c) :=
  pid (Geometry.p2 p)-ch h c p*BitVec.ofNat 64 h.shape.groups
def m (h : Host.Call) (c : Host.Config) (p : Geometry.PID h c) :=
  (pid (Geometry.p0 p)).sdiv (ceil h.shape.chunk c.N)
def n (h : Host.Call) (c : Host.Config) (p : Geometry.PID h c) :=
  (pid (Geometry.p0 p)).srem (ceil h.shape.chunk c.N)

theorem controls (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (metadata : Metadata h c) (p : Geometry.PID h c) :
    b h c p=BitVec.ofNat 64 (Geometry.p1 p) ∧
    ch h c p=BitVec.ofNat 64 (Geometry.p2 p/h.shape.groups) ∧
    head h c p=BitVec.ofNat 64 (Geometry.p2 p-Geometry.p2 p/h.shape.groups*h.shape.groups) ∧
    m h c p=BitVec.ofNat 64 (Geometry.p0 p/Host.cdiv h.shape.chunk c.N) ∧
    n h c p=BitVec.ofNat 64 (Geometry.p0 p%Host.cdiv h.shape.chunk c.N) ∧
    ceil h.shape.K c.K=BitVec.ofNat 64 (Host.cdiv h.shape.K c.K) := by
  have hp0 : Geometry.p0 p<2147483648 := by
    have ht := (p ⟨0,by change 0<3; decide⟩).isLt
    change Geometry.p0 p<Host.cdiv h.shape.chunk c.M*Host.cdiv h.shape.chunk c.N at ht
    have := metadata.gridX; omega
  have hp1 : Geometry.p1 p<2147483648 := by
    have ht := (p ⟨1,by change 1<3; decide⟩).isLt
    change Geometry.p1 p<h.shape.batch at ht
    have := metadata.gridY; omega
  have hp2 : Geometry.p2 p<2147483648 := by
    have := Geometry.p2_lt p; have := metadata.gridZ; omega
  obtain ⟨_,hchunks,hgroups,_⟩ := Finite.positive_dimensions h.shape metadata.sizes
  have hg : h.shape.groups≤65535 :=
    (Nat.le_mul_of_pos_left h.shape.groups hchunks).trans metadata.gridZ
  obtain ⟨_,hchunk,hK⟩ := Finite.dimensions_bounded h.shape metadata.sizes
  obtain ⟨_,hn,hk⟩ := Host.configuration_positive c hc
  obtain ⟨_,hnSmall,hkSmall⟩ := Finite.configuration_small c hc
  have hceil : ceil h.shape.chunk c.N=BitVec.ofNat 64 (Host.cdiv h.shape.chunk c.N) :=
    ceil_exact _ _ hn (by omega)
  have htiles : Host.cdiv h.shape.chunk c.N<9223372036854775808 := by
    unfold Host.cdiv
    have := Nat.div_le_self (h.shape.chunk+c.N-1) c.N
    omega
  have hch : ch h c p=BitVec.ofNat 64 (Geometry.p2 p/h.shape.groups) := by
    rw [ch,pid_exact _ hp2,division _ _ (by omega) (by omega)]
  refine ⟨pid_exact _ hp1,hch,?_,?_,?_,ceil_exact _ _ hk (by omega)⟩
  · rw [head,pid_exact _ hp2,hch,←BitVec.ofNat_mul]
    have hp := Nat.div_mul_le_self (Geometry.p2 p) h.shape.groups
    exact BitVec.ofNat_sub_ofNat_of_le _ _ (by omega) hp
  · rw [m,pid_exact _ hp0,hceil,division _ _ (by omega) htiles]
  · rw [n,pid_exact _ hp0,hceil,remainder _ _ (by omega) htiles]

#axiomsClean controls
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeControl

#guard_msgs (drop info) in
#auditModuleAxioms
