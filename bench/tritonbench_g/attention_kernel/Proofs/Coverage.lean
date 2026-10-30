import bench.tritonbench_g.attention_kernel.Proofs.Grid
import bench.tritonbench_g.attention_kernel.Proofs.Domain
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Whole
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 4000000

def grid (p : Parameters) : Grid := ⟨(p.length+p.bm-1)/p.bm,p.batches*p.heads,1⟩

theorem separated (p : Parameters) (storage : TensorStorage p)
    (live : 0<p.batches ∧ 0<p.heads ∧ 0<p.length) : Separated p :=
  ⟨Ne.symm (storage.separated live).1,Ne.symm (storage.separated live).2.1,
   Ne.symm (storage.separated live).2.2.1,Ne.symm (storage.separated live).2.2.2⟩

def lane (p : Parameters) (hbm : 0<p.bm) (n d : Nat) (hd : d<p.dim) : TileIndex [p.bm,p.dim] :=
  (⟨n%p.bm,Nat.mod_lt _ hbm⟩,⟨d,hd⟩,())

theorem grid_covers (p : Parameters) (hbm : 0<p.bm) (b h n d : Nat)
    (hb : b<p.batches) (hh : h<p.heads) (hn : n<p.length) (hd : d<p.dim) :
    (n/p.bm,b*p.heads+h,0)∈programs (grid p) ∧
    (n/p.bm)*p.bm+n%p.bm=n := by
  refine ⟨?_,(by simpa only [Nat.mul_comm] using Nat.div_add_mod n p.bm)⟩
  rw [mem_programs]
  change n/p.bm<(p.length+p.bm-1)/p.bm ∧ b*p.heads+h<p.batches*p.heads ∧ 0<1
  have hm:=Nat.mul_le_mul_right p.heads (show b+1≤p.batches by omega)
  rw [Nat.add_mul,Nat.one_mul] at hm
  exact ⟨row_program_bound p.length p.bm n hbm hn,by omega,by decide⟩

variable {α : Type} [Inhabited α]

theorem grid_unique (p : Parameters) (guard : Guard p) (s : State α) :
    Unique p (grid p) (programs (grid p)) s := by
  intro a ha b hb i j hi hj he
  rcases a with ⟨ax,ay,az⟩
  rcases b with ⟨bx,byy,bz⟩
  obtain ⟨hax,hay,haz⟩:=(mem_programs (grid p) ax ay az).mp ha
  obtain ⟨hbx,hby,hbz⟩:=(mem_programs (grid p) bx byy bz).mp hb
  have ha0 : az=0 := by change az<1 at haz;omega
  have hb0 : bz=0 := by change bz<1 at hbz;omega
  obtain ⟨hH,haB,haH⟩:=head_decode p ay hay
  obtain ⟨_,hbB,hbH⟩:=head_decode p byy hby
  change ax*p.bm+i.1.val<p.length at hi
  change bx*p.bm+j.1.val<p.length at hj
  change ay/p.heads*p.o.batch+ay%p.heads*p.o.head+(ax*p.bm+i.1.val)*p.o.row+i.2.1.val*p.o.dim=
    byy/p.heads*p.o.batch+byy%p.heads*p.o.head+(bx*p.bm+j.1.val)*p.o.row+j.2.1.val*p.o.dim at he
  obtain ⟨hz,hh,hn,hd⟩:=dense_injective p p.o (guard.dense p.o (by simp))
    (ay/p.heads) (ay%p.heads) (ax*p.bm+i.1.val) i.2.1.val
    (byy/p.heads) (byy%p.heads) (bx*p.bm+j.1.val) j.2.1.val
    haH hi i.2.1.isLt hbH hj j.2.1.isLt he
  have hy : ay=byy := by
    have h1:=Nat.div_add_mod ay p.heads
    have h2:=Nat.div_add_mod byy p.heads
    rw [hz,hh] at h1
    omega
  obtain ⟨hx,hr⟩:=block_unique p.bm ax bx i.1.val j.1.val i.1.isLt j.1.isLt hn
  refine ⟨by simp [hx,hy,ha0,hb0],?_⟩
  rcases i with ⟨ir,id,⟨⟩⟩
  rcases j with ⟨jr,jd,⟨⟩⟩
  simp only [Prod.mk.injEq,Fin.mk.injEq,true_and,and_true]
  exact ⟨Fin.ext hr,Fin.ext hd⟩

#axiomsClean grid_covers
#axiomsClean grid_unique
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Whole
