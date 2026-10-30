import bench.tritonbench_g.streamk_matmul.Proofs.Protocol
import bench.tritonbench_g.int_scaled_matmul.Proofs.Geometry
namespace VeriTile.Bench.Repairs.StreamK.Whole001
open VeriTile.Bench.TritonBenchG.StreamK
namespace Geometry
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry
set_option maxHeartbeats 6000000

/-- PyTorch contiguous strides retain their actual supplied values when a
singleton dimension makes that stride irrelevant, or either dimension is empty. -/
theorem contiguous_empty (rows cols sr sc : Nat) (h : rows=0 ∨ cols=0) :
    Contiguous rows cols sr sc := by
  constructor <;> intros <;> omega

theorem packed_injective (rows cols : Nat) :
    Function.Injective (fun x : Fin rows × Fin cols => x.1.val*cols+x.2.val) := by
  rintro ⟨r,c⟩ ⟨s,d⟩ he
  dsimp only at he
  have hc:=c.isLt;have hd:=d.isLt
  have hcols:0 < cols:=by omega
  have hleft:(r.val*cols+c.val)/cols=r.val:=Nat.div_eq_of_lt_le (by omega)
    (by rw [Nat.add_mul,Nat.one_mul]; omega)
  have hright:(s.val*cols+d.val)/cols=s.val:=Nat.div_eq_of_lt_le (by omega)
    (by rw [Nat.add_mul,Nat.one_mul]; omega)
  have hrs:r.val=s.val:=by rw [←hleft,←hright,he]
  have hcd:c.val=d.val:=by simp only [hrs] at he; omega
  exact Prod.ext (Fin.ext hrs) (Fin.ext hcd)

def layout (h : HostInput) : Layout := ⟨h.m,h.n,h.bm,h.bn,8,h.n,1⟩

theorem valid (h : HostInput) (hM : 0 < h.bm) (hN : 0 < h.bn) : Valid (layout h) := by
  refine ⟨hM,hN,by change 0 < 8; decide,?_⟩
  intro x y he
  apply packed_injective h.m h.n
  change x.1.val*h.n+x.2.val*1=y.1.val*h.n+y.2.val*1 at he
  simpa only [Nat.mul_one] using he

/-- The exact original grouped mapping, including rotated final groups, covers
each output cell once. Its live store pointer fits the fresh M*N allocation. -/
theorem output_owner (h : HostInput) (hM : 0 < h.bm) (hN : 0 < h.bn)
    (x : Cell (layout h)) :
    ∃! tile, tile < count (layout h) ∧ Owns (layout h) tile x := by
  obtain ⟨tile,ht,ho⟩:=covers (layout h) (valid h hM hN) x
  refine ⟨tile,⟨ht,ho⟩,?_⟩
  intro other hx
  exact owner_unique (layout h) (valid h hM hN) other tile hx.1 ht x hx.2 ho

theorem output_mask_safe (h : HostInput) (tile : Nat) (i : Lane (layout h))
    (hi : Active (layout h) tile i) : offset (layout h) tile i < h.m*h.n := by
  simpa [offset,layout] using packed_bound h.m h.n _ _ hi.1 hi.2

theorem scratch_injective (P BM BN : Nat) :
    Function.Injective (fun x : Fin P × Fin BM × Fin BN =>
      x.1.val*BM*BN+x.2.1.val*BN+x.2.2.val) := by
  intro x y he
  let a:Fin (P*BM):=⟨x.1.val*BM+x.2.1.val,packed_bound _ _ _ _ x.1.isLt x.2.1.isLt⟩
  let b:Fin (P*BM):=⟨y.1.val*BM+y.2.1.val,packed_bound _ _ _ _ y.1.isLt y.2.1.isLt⟩
  have he':a.val*BN+x.2.2.val=b.val*BN+y.2.2.val:=by
    dsimp [a,b]; simpa only [Nat.add_mul,Nat.add_assoc] using he
  have hab:=@packed_injective (P*BM) BN (a,x.2.2) (b,y.2.2) he'
  have heab:=congrArg (fun z : Fin (P*BM) × Fin BN => z.1.val) hab
  have hpair:=@packed_injective P BM (x.1,x.2.1) (y.1,y.2.1) heab
  have h1:x.1=y.1:=congrArg (fun z : Fin P × Fin BM => z.1) hpair
  have h2:x.2.1=y.2.1:=congrArg (fun z : Fin P × Fin BM => z.2) hpair
  have h3:x.2.2=y.2.2:=congrArg (fun z : Fin (P*BM) × Fin BN => z.2) hab
  exact Prod.ext h1 (Prod.ext h2 h3)

#axiomsClean contiguous_offset
#axiomsClean input_mask_safe
#axiomsClean output_owner
#axiomsClean output_mask_safe
#axiomsClean scratch_bounds
#axiomsClean scratch_injective
end Geometry
end VeriTile.Bench.Repairs.StreamK.Whole001
