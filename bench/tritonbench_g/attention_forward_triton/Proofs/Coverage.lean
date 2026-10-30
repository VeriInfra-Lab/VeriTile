import bench.tritonbench_g.attention_forward_triton.Proofs.Grid
import bench.tritonbench_g.attention_forward_triton.Proofs.Domain
namespace VeriTile.Bench.Repairs.AttentionForwardTriton.Whole
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 4000000

def grid (p : Parameters) : Grid := ⟨(p.length+127)/128,p.batches*p.heads,1⟩

theorem separated (p : Parameters) (storage : TensorStorage p) : Separated p :=
  ⟨Ne.symm storage.separated.1,Ne.symm storage.separated.2.1,
   Ne.symm storage.separated.2.2.1,Ne.symm storage.separated.2.2.2.1,
   Ne.symm storage.separated.2.2.2.2⟩

def lane (n d : Nat) (hd : d<96) : TileIndex [128,128] :=
  (⟨n%128,Nat.mod_lt _ (by decide)⟩,⟨d,by omega⟩,())

theorem grid_covers (p : Parameters) (b h n d : Nat)
    (hb : b<p.batches) (hh : h<p.heads) (hn : n<p.length) (hd : d<96) :
    (n/128,b*p.heads+h,0)∈programs (grid p) ∧
    (n/128)*128+n%128=n := by
  refine ⟨?_,by omega⟩
  rw [mem_programs]
  change n/128<(p.length+127)/128 ∧ b*p.heads+h<p.batches*p.heads ∧ 0<1
  have hm:=Nat.mul_le_mul_right p.heads (show b+1≤p.batches by omega)
  rw [Nat.add_mul,Nat.one_mul] at hm
  exact ⟨by omega,by omega,by decide⟩

variable {α : Type} [Inhabited α]

theorem grid_unique (p : Parameters) (storage : TensorStorage p) (s : State α) :
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
  change ax*128+i.1.val<p.length ∧ i.2.1.val<96 at hi
  change bx*128+j.1.val<p.length ∧ j.2.1.val<96 at hj
  change ay/p.heads*p.o.batch+ay%p.heads*p.o.head+(ax*128+i.1.val)*p.o.row+i.2.1.val*p.o.dim=
    byy/p.heads*p.o.batch+byy%p.heads*p.o.head+(bx*128+j.1.val)*p.o.row+j.2.1.val*p.o.dim at he
  obtain ⟨hz,hh,hn,hd⟩:=storage.outputInjective _ haB _ haH _ hi.1 _ hi.2
    _ hbB _ hbH _ hj.1 _ hj.2 he
  have hy : ay=byy := by
    have h1:=Nat.div_add_mod ay p.heads
    have h2:=Nat.div_add_mod byy p.heads
    rw [hz,hh] at h1
    omega
  have hx : ax=bx := by have:=i.1.isLt;have:=j.1.isLt;omega
  have hr : i.1.val=j.1.val := by omega
  refine ⟨by simp [hx,hy,ha0,hb0],?_⟩
  rcases i with ⟨ir,id,⟨⟩⟩
  rcases j with ⟨jr,jd,⟨⟩⟩
  simp only [Prod.mk.injEq,Fin.mk.injEq,true_and,and_true]
  exact ⟨Fin.ext hr,Fin.ext hd⟩

#axiomsClean grid_covers
#axiomsClean grid_unique
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.Whole
