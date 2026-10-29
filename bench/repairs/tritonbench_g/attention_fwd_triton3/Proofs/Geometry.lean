import bench.repairs.tritonbench_g.attention_fwd_triton3.SourceDefinitions
import bench.tritonbench_g.attention_fwd_triton3.Proofs.MathematicalSpecification
import VeriTile.Triton.Launch.Composition

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Geometry
open VeriTile.Triton

def Strides.at (s : Strides) (z h r d : Nat) : Nat :=
  z*s.batch+h*s.head+r*s.row+d*s.dim

def Strides.span (s : Strides) (Z H N D : Nat) : Nat :=
  if Z=0 ∨ H=0 ∨ N=0 ∨ D=0 then 0
  else (Z-1)*s.batch+(H-1)*s.head+(N-1)*s.row+(D-1)*s.dim+1

theorem address_bound (s : Strides) (z h r d Z H N D : Nat)
    (hz : z<Z) (hh : h<H) (hr : r<N) (hd : d<D) :
    s.at z h r d < s.span Z H N D := by
  have h1:=Nat.mul_le_mul_right s.batch (show z≤Z-1 by omega)
  have h2:=Nat.mul_le_mul_right s.head (show h≤H-1 by omega)
  have h3:=Nat.mul_le_mul_right s.row (show r≤N-1 by omega)
  have h4:=Nat.mul_le_mul_right s.dim (show d≤D-1 by omega)
  have hn : ¬(Z=0 ∨ H=0 ∨ N=0 ∨ D=0) := by omega
  simp only [Strides.at,Strides.span,if_neg hn]
  omega

structure Ordered (s : Strides) (H R D : Nat) : Prop where
  dim_positive : 0<s.dim
  row_stride : D*s.dim≤s.row
  head_stride : R*s.row≤s.head
  batch_stride : H*s.head≤s.batch

theorem lower_tail (s : Strides) (R D r d : Nat)
    (hdp : 0<s.dim) (hrow : D*s.dim≤s.row)
    (hr : r<R) (hd : d<D) : r*s.row+d*s.dim<R*s.row := by
  have ht : d*s.dim<s.row := lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right hd hdp) hrow
  have hm:=Nat.mul_le_mul_right s.row (show r+1≤R by omega)
  nlinarith

theorem chunk_unique {x y a b s : Nat} (ha : a<s) (hb : b<s)
    (he : x*s+a=y*s+b) : x=y ∧ a=b := by
  have hs : 0<s := by omega
  have hdiv := congrArg (fun n=>n/s) he
  have hx : (x*s+a)/s=x := by simp [Nat.mul_comm x s, Nat.mul_add_div hs, Nat.div_eq_of_lt ha]
  have hy : (y*s+b)/s=y := by simp [Nat.mul_comm y s, Nat.mul_add_div hs, Nat.div_eq_of_lt hb]
  dsimp only at hdiv
  rw [hx,hy] at hdiv
  exact ⟨hdiv,by rw [hdiv] at he; omega⟩

theorem address_injective (s : Strides) (H R D : Nat) (ho : Ordered s H R D)
    (z h r d z' h' r' d' : Nat)
    (hh : h<H) (hr : r<R) (hd : d<D)
    (hh' : h'<H) (hr' : r'<R) (hd' : d'<D)
    (he : s.at z h r d=s.at z' h' r' d') :
    z=z' ∧ h=h' ∧ r=r' ∧ d=d' := by
  have ht:=lower_tail s R D r d ho.dim_positive ho.row_stride hr hd
  have ht':=lower_tail s R D r' d' ho.dim_positive ho.row_stride hr' hd'
  have ht1 : r*s.row+d*s.dim<s.head := lt_of_lt_of_le ht ho.head_stride
  have ht2 : r'*s.row+d'*s.dim<s.head := lt_of_lt_of_le ht' ho.head_stride
  have hb : h*s.head+(r*s.row+d*s.dim)<s.batch := by
    have hm:=Nat.mul_le_mul_right s.head (show h+1≤H by omega)
    have hhB:=ho.batch_stride
    nlinarith
  have hb' : h'*s.head+(r'*s.row+d'*s.dim)<s.batch := by
    have hm:=Nat.mul_le_mul_right s.head (show h'+1≤H by omega)
    have hhB:=ho.batch_stride
    nlinarith
  have hz := chunk_unique hb hb' (by simpa [Strides.at,Nat.add_assoc] using he)
  have hhx:=chunk_unique ht1 ht2 hz.2
  have hrd : d*s.dim<s.row := lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right hd ho.dim_positive) ho.row_stride
  have hrd' : d'*s.dim<s.row := lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right hd' ho.dim_positive) ho.row_stride
  have hrx:=chunk_unique hrd hrd' hhx.2
  exact ⟨hz.1,hhx.1,hrx.1,Nat.eq_of_mul_eq_mul_right ho.dim_positive hrx.2⟩

theorem row_coverage (n b r : Nat) (hb : 0<b) (hr : r<n) :
    r/b<tiles n b ∧ r%b<b ∧ r/b*b+r%b=r := by
  have hm:=Nat.mod_lt r hb
  have he:=Nat.div_add_mod r b
  have hdiv : r/b+1≤(n+b-1)/b := by
    apply (Nat.le_div_iff_mul_le hb).mpr
    calc
      (r/b+1)*b = r/b*b+b := by ring
      _ ≤ r+b := Nat.add_le_add_right (Nat.div_mul_le_self r b) b
      _ ≤ n+b-1 := by omega
  exact ⟨by unfold tiles; omega,hm,by simpa [Nat.mul_comm] using he⟩

theorem tile_rows_bounded (n b rounded : Nat) (hb : 0<b)
    (hn : n≤rounded) (hd : b∣rounded) (p i : Nat)
    (hp : p<tiles n b) (hi : i<b) : p*b+i<rounded := by
  obtain ⟨k,hk⟩:=hd
  have ht : tiles n b≤k := by
    unfold tiles
    apply (Nat.div_le_iff_le_mul_add_pred hb).mpr
    subst rounded
    omega
  have hm:=Nat.mul_le_mul_right b (show p+1≤k by omega)
  nlinarith

/-- The allocation uses multiples of 64 even after the persistent fallback
selects a smaller block. No assumption that the last tile fills the allocation. -/
theorem rounded_divisible (n b : Nat) (h : b∣64) : b∣tiles n 64*64 :=
  dvd_mul_of_dvd_right h _

theorem rounded_covers (n : Nat) : n≤tiles n 64*64 := by
  unfold tiles
  omega

#print axioms address_injective
#print axioms tile_rows_bounded
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Geometry
