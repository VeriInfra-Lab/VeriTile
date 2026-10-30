import bench.tritonbench_g.attention_forward_triton.IO
import bench.tritonbench_g.attention_forward_triton.Proofs.SourceSafety
import bench.tritonbench_g.attention_forward_triton.Proofs.ReadCongruence

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.AttentionIO
open VeriTile Triton FP.Structural Source Whole
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def Occurs (p : Parameters) (r : RegionName) (off : Nat) : Prop :=
  ∃ pos : InputPosition p, (inputChannels p)[pos.1].buf=r ∧ inputAddress p pos.1 pos.2=off

theorem read_extension {A B : Type} [Inhabited B] (zero : A) (encode : A → B)
    (p : Parameters) (vals : Inputs A p) (s : State B)
    (bound : ∀ i j, (s.mem (inputChannels p)[i].buf (inputAddress p i j)).read .real=
      encode (payload p vals i j)) (r : RegionName) (off : Nat) (occurs : Occurs p r off) :
    (s.mem r off).read .real=encode (logicalRead zero p vals r off) := by
  unfold Occurs at occurs
  unfold logicalRead
  rw [dif_pos occurs]
  obtain ⟨hr,ho⟩ := Classical.choose_spec occurs
  simpa only [hr,ho] using bound (Classical.choose occurs).1 (Classical.choose occurs).2

theorem real_read_binding (p : Parameters) (work : List Program) (vals : Inputs Real.R p)
    (s : State Real.R) (bound : (realIO p work).InputBinding vals s) :
    ∀ i j, (s.mem (inputChannels p)[i].buf (inputAddress p i j)).read .real=payload p vals i j := by
  intro i j
  fin_cases i <;> simpa [realIO,memory,inputChannels,payload,mapValue] using bound _ j trivial

theorem native_read_binding (I : Native.Instructions) (p : Parameters) (work : List Program)
    (vals : Inputs Native.Word p) (s : State Native.Expr) (bound : (nativeIO I p work).InputBinding vals s) :
    ∀ i j, (s.mem (inputChannels p)[i].buf (inputAddress p i j)).read .real=wordEncode (payload p vals i j) := by
  intro i j
  fin_cases i <;> simpa [nativeIO,memory,inputChannels,payload,mapValue] using bound _ j trivial

theorem occurs_query (p : Parameters) (s : State A) (pid : Program)
    (legal : pid∈programs (grid p)) (i : TileIndex [128,128])
    (active : Source.active p (start (grid p) pid s) i) :
    Occurs p p.Q (base p (start (grid p) pid s) p.q+
      ((start (grid p) pid s).pids 0*128+i.1.val)*p.q.row+i.2.1.val*p.q.dim) := by
  rcases pid with ⟨x,y,z⟩
  obtain ⟨hx,hy,hz⟩ := (mem_programs (grid p) x y z).mp legal
  obtain ⟨_,hb,hh⟩ := head_decode p y hy
  change x*128+i.1.val<p.length ∧ i.2.1.val<96 at active
  exact ⟨⟨⟨0,by simp [inputChannels]⟩,
    (⟨y/p.heads,hb⟩,⟨y%p.heads,hh⟩,⟨x*128+i.1.val,active.1⟩,⟨i.2.1.val,active.2⟩,())⟩,rfl,rfl⟩

theorem occurs_queryScale (p : Parameters) (s : State A) (pid : Program)
    (legal : pid∈programs (grid p)) :
    Occurs p p.QS ((start (grid p) pid s).pids 1*((p.length+127)/128)+
      (start (grid p) pid s).pids 0) := by
  rcases pid with ⟨x,y,z⟩
  obtain ⟨hx,hy,hz⟩ := (mem_programs (grid p) x y z).mp legal
  obtain ⟨_,hb,hh⟩ := head_decode p y hy
  refine ⟨⟨⟨3,by simp [inputChannels]⟩,(⟨y/p.heads,hb⟩,⟨y%p.heads,hh⟩,⟨x,hx⟩,())⟩,rfl,?_⟩
  have h : y/p.heads*p.heads+y%p.heads=y := by
    simpa [Nat.mul_comm] using Nat.div_add_mod y p.heads
  change (y/p.heads*p.heads+y%p.heads)*((p.length+127)/128)+x=
    y*((p.length+127)/128)+x
  rw [h]

theorem occurs_key (p : Parameters) (s : State A) (pid : Program)
    (legal : pid∈programs (grid p)) (row : Nat) (i : TileIndex [128,64])
    (active : i.2.1.val<p.length-row ∧ i.1.val<96) :
    Occurs p p.K (base p (start (grid p) pid s) p.k+i.1.val*p.k.dim+(row+i.2.1.val)*p.k.row) := by
  rcases pid with ⟨x,y,z⟩
  obtain ⟨hx,hy,hz⟩ := (mem_programs (grid p) x y z).mp legal
  obtain ⟨_,hb,hh⟩ := head_decode p y hy
  refine ⟨⟨⟨1,by simp [inputChannels]⟩,
    (⟨y/p.heads,hb⟩,⟨y%p.heads,hh⟩,⟨row+i.2.1.val,by omega⟩,⟨i.1.val,active.2⟩,())⟩,rfl,?_⟩
  change y/p.heads*p.k.batch+y%p.heads*p.k.head+(row+i.2.1.val)*p.k.row+i.1.val*p.k.dim=
    y/p.heads*p.k.batch+y%p.heads*p.k.head+i.1.val*p.k.dim+(row+i.2.1.val)*p.k.row
  omega

theorem occurs_keyScale (p : Parameters) (s : State A) (pid : Program)
    (legal : pid∈programs (grid p)) (row : Nat) (active : row<p.length) :
    Occurs p p.KS ((start (grid p) pid s).pids 1*((p.length+63)/64)+row/64) := by
  rcases pid with ⟨x,y,z⟩
  obtain ⟨hx,hy,hz⟩ := (mem_programs (grid p) x y z).mp legal
  obtain ⟨_,hb,hh⟩ := head_decode p y hy
  refine ⟨⟨⟨4,by simp [inputChannels]⟩,(⟨y/p.heads,hb⟩,⟨y%p.heads,hh⟩,⟨row/64,by omega⟩,())⟩,rfl,?_⟩
  have h : y/p.heads*p.heads+y%p.heads=y := by
    simpa [Nat.mul_comm] using Nat.div_add_mod y p.heads
  change (y/p.heads*p.heads+y%p.heads)*((p.length+63)/64)+row/64=
    y*((p.length+63)/64)+row/64
  rw [h]

theorem occurs_value (p : Parameters) (s : State A) (pid : Program)
    (legal : pid∈programs (grid p)) (row : Nat) (i : TileIndex [64,128])
    (active : i.1.val<p.length-row ∧ i.2.1.val<96) :
    Occurs p p.V (base p (start (grid p) pid s) p.v+(row+i.1.val)*p.v.row+i.2.1.val*p.v.dim) := by
  rcases pid with ⟨x,y,z⟩
  obtain ⟨hx,hy,hz⟩ := (mem_programs (grid p) x y z).mp legal
  obtain ⟨_,hb,hh⟩ := head_decode p y hy
  exact ⟨⟨⟨2,by simp [inputChannels]⟩,
    (⟨y/p.heads,hb⟩,⟨y%p.heads,hh⟩,⟨row+i.1.val,by omega⟩,⟨i.2.1.val,active.2⟩,())⟩,rfl,rfl⟩

/-- Finite logical bindings give precisely the read equality needed by the
existing source recurrence, without constraining any exterior memory cell. -/
theorem reads_of_occurrences {A : Type} [Inhabited A] (p : Parameters)
    (s t : State A) (pid : Program) (legal : pid∈programs (grid p))
    (bound : ∀ r off, Occurs p r off → (s.mem r off).read .real=(t.mem r off).read .real) :
    Logical.Reads p (start (grid p) pid s) (start (grid p) pid t) := by
  refine ⟨rfl,?_,?_,?_,?_,?_⟩
  · intro i active
    exact bound _ _ (occurs_query p s pid legal i active)
  · exact bound _ _ (occurs_queryScale p s pid legal)
  · intro row _ i active
    exact bound _ _ (occurs_key p s pid legal row i active)
  · intro row active
    exact bound _ _ (occurs_keyScale p s pid legal row active)
  · intro row _ i active
    exact bound _ _ (occurs_value p s pid legal row i active)

end
end VeriTile.Bench.Repairs.AttentionForwardTriton.AttentionIO
