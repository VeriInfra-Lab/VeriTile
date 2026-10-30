import bench.tritonbench_g.mixed_sparse_attention.Proofs.Grid
import bench.tritonbench_g.mixed_sparse_attention.Proofs.Domain
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Whole
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 4000000

def grid (c : Config) : Grid := ⟨(c.T+c.BM-1)/c.BM,c.Z*c.H,1⟩
def logicalAddress (c : Config) (hz n d : Nat) : Nat := (hz*c.T+n)*c.BD+d

def LengthsBound {α : Type} [Inhabited α] (c : Config) (s : State α) : Prop :=
  ∀ b<c.Z,(s.mem (Region.cast c.seq) b).read .nat≤c.T

def lane (c : Config) (hbm : 0<c.BM) (n d : Nat) (hd : d<c.BD) : TileIndex [c.BM,c.BD] :=
  (⟨n%c.BM,Nat.mod_lt _ hbm⟩,⟨d,hd⟩,())

theorem grid_covers (c : Config) (hbm : 0<c.BM) (hz n d : Nat)
    (hhz : hz<c.Z*c.H) (hn : n<c.T) (hd : d<c.BD) :
    (n/c.BM,hz,0)∈programs (grid c) ∧ (n/c.BM)*c.BM+n%c.BM=n := by
  refine ⟨?_,by simpa only [Nat.mul_comm] using Nat.div_add_mod n c.BM⟩
  rw [mem_programs]
  exact ⟨row_program_bound c.T c.BM n hbm hn,hhz,by change 0<1;omega⟩

variable {α : Type} [Inhabited α]

theorem source_address (c : Config) (layout : Contiguous c) (g : Grid) (p : Program)
    (s : State α) (i : TileIndex [c.BM,c.BD]) :
    address c (start g p s) i=logicalAddress c p.2.1 (p.1*c.BM+i.1.val) i.2.1.val := by
  exact contiguous_address c layout (start g p s) i

theorem active_row_bound (c : Config) (s : State α) (hl : LengthsBound c s) (p : Program)
    (hp : p∈programs (grid c)) (i : TileIndex [c.BM,c.BD]) (hi : active c (start (grid c) p s) i) :
    p.1*c.BM+i.1.val<c.T := by
  obtain ⟨_,hhz,_⟩:=(mem_programs (grid c) p.1 p.2.1 p.2.2).mp hp
  have hH : 0<c.H := by
    change p.2.1<c.Z*c.H at hhz
    by_contra hn
    have he : c.H=0 := by omega
    simp [he] at hhz
  have hb : p.2.1/c.H<c.Z := (Nat.div_lt_iff_lt_mul hH).mpr hhz
  exact lt_of_lt_of_le hi (hl _ hb)

theorem grid_unique (c : Config) (layout : Contiguous c) (s : State α) (hl : LengthsBound c s) :
    Unique c (grid c) (programs (grid c)) s := by
  intro a ha b hb i j hi hj he
  have hn:=active_row_bound c s hl a ha i hi
  have hn':=active_row_bound c s hl b hb j hj
  rw [source_address c layout,source_address c layout] at he
  obtain ⟨hrow,hd⟩:=block_unique c.BD _ _ _ _ i.2.1.isLt j.2.1.isLt he
  obtain ⟨hy,hr⟩:=block_unique c.T _ _ _ _ hn hn' hrow
  obtain ⟨hx,hidx⟩:=block_unique c.BM _ _ _ _ i.1.isLt j.1.isLt hr
  have hz : a.2.2=0 := by have hh:=((mem_programs (grid c) a.1 a.2.1 a.2.2).mp ha).2.2;change a.2.2<1 at hh;omega
  have hz' : b.2.2=0 := by have hh:=((mem_programs (grid c) b.1 b.2.1 b.2.2).mp hb).2.2;change b.2.2<1 at hh;omega
  refine ⟨?_,?_⟩
  · rcases a with ⟨ax,ay,az⟩
    rcases b with ⟨bx,byy,bz⟩
    simp_all
  · rcases i with ⟨ir,id,⟨⟩⟩
    rcases j with ⟨jr,jd,⟨⟩⟩
    exact Prod.ext (Fin.ext hidx) (Prod.ext (Fin.ext hd) rfl)

#axiomsClean grid_covers
#axiomsClean source_address
#axiomsClean grid_unique
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Whole
