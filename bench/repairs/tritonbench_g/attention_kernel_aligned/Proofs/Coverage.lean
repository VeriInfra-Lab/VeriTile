import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Grid
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Domain
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Whole
open VeriTile Triton FP.Structural Source Full
noncomputable section
set_option maxHeartbeats 4000000

def grid (p : Parameters) : Grid := ⟨(p.length+p.blockM-1)/p.blockM,p.batches*p.heads,1⟩

theorem separated (p : Parameters) (storage : TensorStorage p) (ne : Nonempty p) : Separated p :=
  ⟨Ne.symm (storage.separated ne).1,Ne.symm (storage.separated ne).2.1,
   Ne.symm (storage.separated ne).2.2.1,Ne.symm (storage.separated ne).2.2.2⟩

def lane (p : Parameters) (selected : Selected p) (n d : Nat) (hd : d<p.head) : QLane p :=
  (⟨n%p.blockM,Nat.mod_lt _ (selected.blockM_pos p)⟩,⟨d,hd⟩,())

theorem grid_covers (p : Parameters) (selected : Selected p) (b h n : Nat)
    (hb : b<p.batches) (hh : h<p.heads) (hn : n<p.length) :
    (n/p.blockM,b*p.heads+h,0)∈programs (grid p) ∧ (n/p.blockM)*p.blockM+n%p.blockM=n := by
  refine ⟨?_,by have:=Nat.div_add_mod' n p.blockM; omega⟩
  rw [mem_programs]
  change n/p.blockM<(p.length+p.blockM-1)/p.blockM ∧ b*p.heads+h<p.batches*p.heads ∧ 0<1
  have hm:=Nat.mul_le_mul_right p.heads (show b+1≤p.batches by omega)
  rw [Nat.add_mul,Nat.one_mul] at hm
  refine ⟨?_,by omega,by decide⟩
  have hb:=selected.blockM
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hb
  rcases hb with hb|hb|hb|hb <;> rw [hb] <;> omega

variable {α : Type} [Inhabited α]

theorem grid_unique (p : Parameters) (g : Guard p) (s : State α) :
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
  change ax*p.blockM+i.1.val<p.length at hi
  change bx*p.blockM+j.1.val<p.length at hj
  change offset (oStrides p) (ay/p.heads) (ay%p.heads) (ax*p.blockM+i.1.val) i.2.1.val=
    offset (oStrides p) (byy/p.heads) (byy%p.heads) (bx*p.blockM+j.1.val) j.2.1.val at he
  have ne : Nonempty p := ⟨Nat.zero_lt_of_lt haB,hH,Nat.zero_lt_of_lt hi⟩
  obtain ⟨hz,hh,hn,hd⟩:=outputInjective p (g.layout ne) _ _ _ _ _ _ _ _ haB haH hi i.2.1.isLt
    hbB hbH hj j.2.1.isLt he
  have hy : ay=byy := by
    have h1:=Nat.div_add_mod ay p.heads
    have h2:=Nat.div_add_mod byy p.heads
    rw [hz,hh] at h1
    omega
  have hx : ax=bx := by
    have hiLt:=i.1.isLt
    have hjLt:=j.1.isLt
    have hb:=g.selected.blockM
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hb
    rcases hb with hb|hb|hb|hb <;> simp only [hb] at hn hiLt hjLt <;> omega
  have hr : i.1.val=j.1.val := by rw [hx] at hn;omega
  refine ⟨by simp [hx,hy,ha0,hb0],?_⟩
  rcases i with ⟨ir,id,⟨⟩⟩
  rcases j with ⟨jr,jd,⟨⟩⟩
  simp only [Prod.mk.injEq,Fin.mk.injEq,true_and,and_true]
  exact ⟨Fin.ext hr,Fin.ext hd⟩

#axiomsClean grid_covers
#axiomsClean grid_unique
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Whole
