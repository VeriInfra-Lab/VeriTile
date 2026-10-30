import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Geometry
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Index
noncomputable section
set_option maxHeartbeats 5000000
/-- Bounds include formed masked lane addresses and the final reverse pointer
decrement. Allocation-relative element arithmetic is signed i32; byte pointers
are i64. The z launch dimension is the actual BH dimension. -/
structure Safe (s : Shape) (f : Flags) : Prop where
  positive : s.Positive
  gridZ : s.BH≤65535
  gridForward : s.FNV<2^31
  gridBackward : s.BNV<2^31
  span : ∀ r,Geometry.capacity s f r+Geometry.rows s r*Geometry.cols s f r+
    Geometry.cols s f r+s.BK*s.V+s.BBV+s.BK<2^31

theorem capacity_safe (s : Shape) (f : Flags) (h : Safe s f) (r : Region) :
    Geometry.capacity s f r<2^31 := by have hh:=h.span r;omega
theorem active_i32 (s : Shape) (f : Flags) (h : Safe s f) (a : Addr) (ha : Geometry.Valid s f a) :
    0≤(Geometry.offset s f a:Int) ∧ (Geometry.offset s f a:Int)<2147483648 := by
  have h1:=Geometry.offset_bound s f a ha
  have h2:=capacity_safe s f h a.region
  constructor <;> omega

theorem i32_exact (x : Int) (lo : -2147483648≤x) (hi : x<2147483648) :
    (BitVec.ofInt 32 x).toInt=x := by
  rw [BitVec.toInt_ofInt]
  apply Int.bmod_eq_of_le <;> norm_num <;> assumption

theorem active_i32_exact (s : Shape) (f : Flags) (h : Safe s f) (a : Addr) (ha : Geometry.Valid s f a) :
    (BitVec.ofInt 32 (Geometry.offset s f a)).toInt=Geometry.offset s f a := by
  have hh:=active_i32 s f h a ha
  exact i32_exact _ (by omega) hh.2
/-- Also covers padded lanes: rows may be one-past after forward, and col
may extend into BK/BV padding. These formed pointers are never dereferenced
without the source masks. -/
theorem formed_bound (P R C pad plane row col : Nat)
    (hp : plane<P) (hr : row≤R) (hc : col<C+pad) :
    (plane*R+row)*C+col<P*R*C+C+pad := by
  have hp' : plane*R+row≤P*R := by
    calc
      _ ≤ plane*R+R := Nat.add_le_add_left hr _
      _ = (plane+1)*R := by rw [Nat.add_mul];simp
      _ ≤ P*R := Nat.mul_le_mul_right R (by omega)
  have hm:=Nat.mul_le_mul_right C hp'
  omega

theorem reverse_pointer_bound (base stride time lane cap : Nat)
    (hspan : base+time*stride+lane<cap) (hcap : cap<2147483648) (hs : stride≤2147483648) :
    -2147483648≤(base:Int)+(time:Int)*stride+lane-stride ∧
    (base:Int)+(time:Int)*stride+lane-stride<2147483648 := by
  have hcast : (base:Int)+(time:Int)*stride+lane<cap := by exact_mod_cast hspan
  constructor <;> nlinarith

theorem active_byte_i64 (s : Shape) (f : Flags) (h : Safe s f) (a : Addr)
    (ha : Geometry.Valid s f a) (byte : Fin 4) :
    0≤4*(Geometry.offset s f a:Int)+byte.val ∧
    4*(Geometry.offset s f a:Int)+byte.val<9223372036854775808 := by
  have hh:=active_i32 s f h a ha
  have hb:=byte.isLt
  constructor <;> omega

theorem byte_address_injective (s : Shape) (f : Flags) (a b : Addr)
    (ha : Geometry.Valid s f a) (hb : Geometry.Valid s f b) (hr : a.region=b.region)
    (i j : Fin 4) (he : 4*Geometry.offset s f a+i.val=4*Geometry.offset s f b+j.val) : a=b ∧ i=j := by
  have hi:=i.isLt
  have hj:=j.isLt
  have hoff : Geometry.offset s f a=Geometry.offset s f b := by omega
  have heq : i=j := Fin.ext (by omega)
  exact ⟨Geometry.offset_injective s f a b ha hb hr hoff,heq⟩

def rowPadding (s : Shape) : Region → Nat
  | .h0 | .ht | .dht | .dh0 => s.BK
  | _ => 0
theorem formed_source_i32 (s : Shape) (f : Flags) (h : Safe s f) (a : Addr)
    (hp : a.plane<Geometry.planes s a.region)
    (hr : a.row≤Geometry.rows s a.region+rowPadding s a.region)
    (hc : a.col<Geometry.cols s f a.region+s.BBV+s.BK) :
    (Geometry.offset s f a:Int)<2147483648 := by
  have hrow : a.plane*Geometry.rows s a.region+a.row≤
      Geometry.planes s a.region*Geometry.rows s a.region+rowPadding s a.region := by
    calc
      _ ≤ (a.plane+1)*Geometry.rows s a.region+rowPadding s a.region := by
        rw [Nat.add_mul];omega
      _ ≤ Geometry.planes s a.region*Geometry.rows s a.region+rowPadding s a.region :=
        Nat.add_le_add_right (Nat.mul_le_mul_right _ (by omega)) _
  have hm:=Nat.mul_le_mul_right (Geometry.cols s f a.region) hrow
  simp only [Nat.add_mul] at hm
  have hpad : rowPadding s a.region*Geometry.cols s f a.region≤s.BK*s.V := by
    cases a.region <;> simp [rowPadding,Geometry.cols]
  have hs:=h.span a.region
  unfold Geometry.offset
  simp only [Nat.add_mul]
  dsimp only [Geometry.capacity] at hs
  omega

#axiomsClean formed_source_i32

#axiomsClean active_i32_exact
#axiomsClean formed_bound
#axiomsClean reverse_pointer_bound
#axiomsClean active_byte_i64
#axiomsClean byte_address_injective
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Index
