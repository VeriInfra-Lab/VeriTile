import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.RawMemory

/-! Kernel-only execution: the output buffer already exists. Inactive rows
retain their entry cells. Every CTA receives fresh IDs and predecessor memory. -/
namespace VeriTile.Bench.Repairs.MixedSparseAttention.KernelGrid
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked
noncomputable section
open Classical
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

def live (c : Config) (s : State α) (q : Raw.Coord c) : Prop :=
  Raw.row q < (s.mem (Region.cast c.seq) (Raw.hz q/c.H)).read .nat
def Writes (c : Config) (s : State α) (r : RegionName) (off : Nat) : Prop :=
  r=c.out ∧ ∃ q : Raw.Coord c,live c s q ∧ off=Raw.offset q
def owner (c : Config) (q : Raw.Coord c) : Program := (Raw.row q/c.BM,Raw.hz q,0)
def rowLane (c : Config) (hbm : 0<c.BM) (q : Raw.Coord c) :=
  lane c hbm (Raw.row q) (Raw.dim q) q.2.2.isLt

theorem owned (c : Config) (layout : Contiguous c) (s : State α) (hbm : 0<c.BM)
    (q : Raw.Coord c) :
    owner c q∈programs (grid c) ∧
    address c (start (grid c) (owner c q) s) (rowLane c hbm q)=Raw.offset q ∧
    (active c (start (grid c) (owner c q) s) (rowLane c hbm q) ↔ live c s q) := by
  have hc:=grid_covers c hbm (Raw.hz q) (Raw.row q) (Raw.dim q) q.1.isLt q.2.1.isLt q.2.2.isLt
  refine ⟨hc.1,?_,?_⟩
  · rw [source_address c layout]
    simpa only [owner,rowLane,lane,Raw.offset,hc.2]
  · simp only [active,rowStart,sequenceLength,start,ids,owner,rowLane,lane,live]
    simpa only [if_true,if_false,show ¬(1:Nat)=0 by decide,hc.2]

theorem write_of_store (c : Config) (layout : Contiguous c) (s : State α)
    (hl : LengthsBound c s) (p : Program) (hp : p∈programs (grid c))
    (i : TileIndex [c.BM,c.BD]) (hi : active c (start (grid c) p s) i) :
    Writes c s c.out (address c (start (grid c) p s) i) := by
  have hh:=((mem_programs (grid c) p.1 p.2.1 p.2.2).mp hp).2.1
  have ht:=active_row_bound c s hl p hp i hi
  let q : Raw.Coord c := (⟨p.2.1,hh⟩,⟨p.1*c.BM+i.1.val,ht⟩,i.2.1)
  refine ⟨rfl,q,?_,?_⟩
  · exact hi
  · exact source_address c layout (grid c) p s i

theorem inactive (c : Config) (s : State α) (q : Raw.Coord c) (h : ¬live c s q) :
    ¬Writes c s c.out (Raw.offset q) := by
  rintro ⟨_,p,hp,he⟩
  have eq:=Raw.offset_injective he
  exact h (eq.symm ▸ hp)

structure Result (M : Program → Algebra α) (D : Program → Dot α)
    (c : Config) (hbm : 0<c.BM) (before final : State α) : Prop where
  output : ∀ q : Raw.Coord c,live c before q →
    final.mem c.out (Raw.offset q)=.mk .real
      (ofFloat .real (Whole.answer (M (owner c q)) (D (owner c q)) c
        (start (grid c) (owner c q) before) (rowLane c hbm q)))
  frame : ∀ r off,¬Writes c before r off → final.mem r off=before.mem r off

theorem source (M : Program → Algebra α) (D : Program → Dot α)
    (c : Config) (sep : Separated c) (layout : Contiguous c) (s : State α)
    (hl : LengthsBound c s) (hbm : 0<c.BM) (hbn : 0<c.BN)
    (work : List Program) (order : work.Perm (programs (grid c))) :
    ∃ final,runPrograms M D c (grid c) work s=some final ∧ Result M D c hbm s final := by
  have uniq : Unique c (grid c) work s := by
    intro a ha b hb i j hi hj he
    exact grid_unique c layout s hl a (order.mem_iff.mp ha) b (order.mem_iff.mp hb) i j hi hj he
  obtain ⟨final,he,hv,hf⟩:=run_correct M D c sep hbn (grid c) work s uniq
  refine ⟨final,he,⟨?_,?_⟩⟩
  · intro q hq
    have o:=owned c layout s hbm q
    have h:=hv (owner c q) (order.mem_iff.mpr o.1) (rowLane c hbm q) (o.2.2.mpr hq)
    simpa only [o.2.1] using h
  · intro r off hn
    apply hf
    by_cases hr : r=c.out
    · subst r
      right
      intro p hp i hi eq
      apply hn
      rw [eq]
      exact write_of_store c layout s hl p (order.mem_iff.mp hp) i hi
    · exact Or.inl hr

theorem no_programs (c : Config) (hbm : 0<c.BM) (empty : c.Z=0 ∨ c.T=0) :
    programs (grid c)=[] := by
  rcases empty with h|h
  · simp [programs,grid,h]
  · have hz : (c.T+c.BM-1)/c.BM=0 := by apply Nat.div_eq_of_lt;omega
    simp [programs,grid,hz]

#axiomsClean owned
#axiomsClean source
#axiomsClean no_programs
#specNonCircular Writes avoiding [runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.KernelGrid
