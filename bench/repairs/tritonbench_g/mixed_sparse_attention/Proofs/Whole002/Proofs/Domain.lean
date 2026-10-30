import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Reference
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Whole
open VeriTile Triton FP.Structural Source
noncomputable section

structure Contiguous (c : Config) : Prop where
  qz : c.sqz=c.H*c.T*c.BD
  qh : c.sqh=c.T*c.BD
  qm : c.sqm=c.BD
  qk : c.sqk=1
  kz : c.skz=c.H*c.T*c.BD
  kh : c.skh=c.T*c.BD
  kn : c.skn=c.BD
  kk : c.skk=1
  vz : c.svz=c.H*c.T*c.BD
  vh : c.svh=c.T*c.BD
  vn : c.svn=c.BD
  vk : c.svk=1
  oz : c.soz=c.H*c.T*c.BD
  oh : c.soh=c.T*c.BD
  om : c.som=c.BD
  ok : c.sok=1

structure Shape (c : Config) : Prop where
  batches : 0<c.Z
  heads : 0<c.H
  bm : c.BM∈[16,32,64,128]
  bn : c.BN∈[16,32,64,128]
  dim : c.BD∈[16,32,64,128]
  rows : c.NR=(c.T+c.BM-1)/c.BM
  contiguous : Contiguous c

theorem ceil_cover (N block : Nat) (hb : 0<block) :
    N≤(N+block-1)/block*block := by
  have rem:=Nat.mod_lt (N+block-1) hb
  have eqn:=Nat.div_add_mod (N+block-1) block
  rw [Nat.mul_comm block] at eqn
  omega

theorem query_lane_bound (N block pid lane : Nat) (hb : 0<block)
    (hp : pid<(N+block-1)/block) (hl : lane<block) : pid*block+lane<N+block := by
  have hprod:=Nat.mul_le_mul_right block (show pid+1≤(N+block-1)/block by omega)
  have hdiv:=Nat.div_mul_le_self (N+block-1) block
  rw [Nat.add_mul,Nat.one_mul] at hprod
  omega

theorem row_program_bound (N block n : Nat) (hb : 0<block) (hn : n<N) :
    n/block<(N+block-1)/block :=
  (Nat.div_lt_iff_lt_mul hb).mpr (lt_of_lt_of_le hn (ceil_cover N block hb))

theorem block_unique (pitch x y i j : Nat) (hi : i<pitch) (hj : j<pitch)
    (he : x*pitch+i=y*pitch+j) : x=y ∧ i=j := by
  have hp : 0<pitch := by omega
  have hx : (x*pitch+i)/pitch=x := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hi]
    omega
  have hy : (y*pitch+j)/pitch=y := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hj]
    omega
  have hxy : x=y := by
    have h:=congrArg (fun n=>n/pitch) he
    simpa only [hx,hy] using h
  exact ⟨hxy,by simpa only [hxy,Nat.add_left_cancel_iff] using he⟩


theorem contiguous_address {α : Type} [Inhabited α] (c : Config) (layout : Contiguous c)
    (s : State α) (i : TileIndex [c.BM,c.BD]) :
    queryBase c s+(rowStart c s+i.1.val)*c.som+i.2.1.val*c.sok=
      (s.pids 1*c.T+(s.pids 0*c.BM+i.1.val))*c.BD+i.2.1.val := by
  rw [queryBase,rowStart,layout.qz,layout.qh,layout.om,layout.ok,Nat.mul_one]
  have hp : s.pids 1/c.H*c.H+s.pids 1%c.H=s.pids 1 := by
    simpa [Nat.mul_comm] using Nat.div_add_mod (s.pids 1) c.H
  calc
    _=((s.pids 1/c.H*c.H+s.pids 1%c.H)*c.T+(s.pids 0*c.BM+i.1.val))*c.BD+i.2.1.val := by ring
    _=_ := by rw [hp]

#axiomsClean contiguous_address
#axiomsClean block_unique
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Whole
