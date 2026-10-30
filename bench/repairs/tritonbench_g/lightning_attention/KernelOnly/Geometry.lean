import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Grid
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Geometry
open VeriTile Triton
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (packed_injective packed_lt)
set_option maxHeartbeats 6000000

def dense (N width head time feature : Nat) := (head*N+time)*width+feature

theorem dense_collision {N width head other time later feature column : Nat}
    (ht : time<N) (hu : later<N) (hf : feature<width) (hg : column<width)
    (he : dense N width head time feature=dense N width other later column) :
    head=other ∧ time=later ∧ feature=column := by
  obtain ⟨hr,hc⟩ := packed_injective hf hg he
  obtain ⟨hh,ht⟩ := packed_injective ht hu hr
  exact ⟨hh,ht,hc⟩

theorem dense_bound {N width heads head time feature : Nat}
    (hh : head<heads) (ht : time<N) (hf : feature<width) :
    dense N width head time feature<heads*N*width := packed_lt (packed_lt hh ht) hf

theorem channels_exact (d : Host.Dimensions) (h : Host.Contract d) : d.NV*d.BM=d.E := by
  have hz := Nat.mod_eq_zero_of_dvd (Host.bm_divides d h)
  rcases Host.bm_legal d h with hb|hb
  · simp only [hb] at hz
    unfold Host.Dimensions.NV
    rw [hb]
    omega
  · simp only [hb] at hz
    unfold Host.Dimensions.NV
    rw [hb]
    omega

theorem channel_active (d : Host.Dimensions) (h : Host.Contract d) (channel : Nat)
    (hc : channel<d.NV) (lane : Fin d.BM) : channel*d.BM+lane.val<d.E := by
  have ht := packed_lt hc lane.isLt
  rwa [channels_exact d h] at ht

theorem pid_ext {a b : PID} (hh : a.head=b.head) (hc : a.channel=b.channel) : a=b := by
  cases a; cases b; simp_all

theorem fwd_address (p : Parameters) (pid : PID) (block : Nat) (i : TileIndex [p.BT,p.BM]) :
    fwdOffset p pid block i=dense p.N p.E pid.head (block*p.BT+i.1.val) (pid.channel*p.BM+i.2.1.val) := by
  unfold fwdOffset dense
  ring

theorem fwd_collision (p : Parameters) (a b : PID) (left right : Nat)
    (i j : TileIndex [p.BT,p.BM]) (hi : fwdActive p left i) (hj : fwdActive p right j)
    (hic : a.channel*p.BM+i.2.1.val<p.E) (hjc : b.channel*p.BM+j.2.1.val<p.E)
    (he : fwdOffset p a left i=fwdOffset p b right j) : a=b ∧ left=right ∧ i=j := by
  rw [fwd_address,fwd_address] at he
  obtain ⟨hh,ht,hc⟩ := dense_collision hi hj hic hjc he
  obtain ⟨hb,hl⟩ := packed_injective i.1.isLt j.1.isLt ht
  obtain ⟨hv,hv'⟩ := packed_injective i.2.1.isLt j.2.1.isLt hc
  exact ⟨pid_ext hh hv,hb,Prod.ext (Fin.ext hl) (Prod.ext (Fin.ext hv') (Subsingleton.elim _ _))⟩

theorem fwd_bound (d : Host.Dimensions) (h : Host.Contract d) (pid : PID)
    (hp : ValidPID .forward d pid) (block : Nat) (i : TileIndex [64,d.BM])
    (hi : fwdActive d.parameters block i) :
    fwdOffset d.parameters pid block i<d.B*d.H*d.N*d.E := by
  rw [fwd_address]
  exact dense_bound hp.1 hi (channel_active d h pid.channel hp.2 i.2.1)

theorem row_address (N width head start rows : Nat) (i : TileIndex [rows,width]) :
    head*N*width+(start+i.1.val)*width+i.2.1.val=dense N width head (start+i.1.val) i.2.1.val := by
  unfold dense
  ring

theorem trans_address (N width head start rows : Nat) (i : TileIndex [width,rows]) :
    head*N*width+(start+i.2.1.val)*width+i.1.val=dense N width head (start+i.2.1.val) i.1.val := by
  unfold dense
  ring

#axiomsClean dense_collision
#axiomsClean channels_exact
#axiomsClean fwd_collision
#axiomsClean fwd_bound
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Geometry
