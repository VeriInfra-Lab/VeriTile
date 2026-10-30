import bench.repairs.tritonbench_g.rope_embedding.Proofs.Spec
namespace VeriTile.Bench.Repairs.RopeEmbedding
open VeriTile.Triton
set_option maxHeartbeats 2000000

abbrev Lane (r : Request) := TileIndex [r.block]
abbrev Location (r : Request) := Nat×Nat×Nat×Bool×Lane r
def row {r : Request} (l : Location r) := l.1
def group {r : Request} (l : Location r) := l.2.1
def head {r : Request} (l : Location r) := l.2.2.1
def side {r : Request} (l : Location r) := l.2.2.2.1
def lane {r : Request} (l : Location r) := l.2.2.2.2
def column {r : Request} (l : Location r) := (lane l).1.val+if side l then r.half else 0
def active {r : Request} (l : Location r) : Prop :=
  row l<r.rows ∧ head l<r.heads ∧ (lane l).1.val<r.half
instance {r : Request} (l : Location r) : Decidable (active l) := inferInstanceAs (Decidable (_∧_∧_))
def schedule (r : Request) : List (Location r) :=
  (List.range r.rows).flatMap fun row => (List.range r.groups).flatMap fun g =>
    (List.range' (g*4) (min (g*4+4) r.heads-g*4)).flatMap fun h =>
      [Bool.false,Bool.true].flatMap fun im => (TileShape.allIndices [r.block]).map fun i => (row,g,h,im,i)
theorem block_covers (r : Request) : r.half≤r.block := by
  unfold Request.block
  split
  · omega
  · exact Nat.le_pow_clog (by decide) _
theorem group_head (r : Request) (g : Nat) (hg : g<r.groups) : g*4<r.heads := by
  unfold Request.groups at hg
  omega
theorem mem_schedule (r : Request) (l : Location r) :
    l∈schedule r ↔ row l<r.rows ∧ group l<r.groups ∧
      group l*4≤head l ∧ head l<min (group l*4+4) r.heads := by
  simp only [schedule,List.mem_flatMap,List.mem_range,List.mem_range',List.mem_map]
  constructor
  · rintro ⟨a,ha,g,hg,h,hh,im,him,i,hi,he⟩
    subst l
    have hg' := group_head r g hg
    have hs : g*4≤min (g*4+4) r.heads := le_min (by omega) (by omega)
    obtain ⟨n,hn,eq⟩ := hh
    simp only [Nat.one_mul] at eq
    dsimp [row,group,head]
    exact ⟨ha,hg,by omega,by omega⟩
  · rintro ⟨hr,hg,hlo,hhi⟩
    have hg' := group_head r (group l) hg
    have hs : group l*4≤min (group l*4+4) r.heads := le_min (by omega) (by omega)
    refine ⟨row l,hr,group l,hg,head l,⟨head l-group l*4,by omega,by omega⟩,
      side l,?_,lane l,TileShape.mem_allIndices _ _,rfl⟩
    cases side l <;> simp
theorem active_column (r : Request) (l : Location r) (ha : active l) : column l<r.dimension := by
  have : 2*r.half≤r.dimension := by unfold Request.half; omega
  unfold active at ha
  unfold column
  split <;> omega
theorem offset_lt (r : Request) (a h c : Nat) (ha : a<r.rows) (hh : h<r.heads) (hc : c<r.dimension) :
    r.offset a h c<r.size := by
  have h1 : a*r.heads+h<r.rows*r.heads := by
    have := Nat.mul_le_mul_right r.heads (show a+1≤r.rows by omega)
    rw [Nat.add_mul,Nat.one_mul] at this
    omega
  calc
    r.offset a h c=(a*r.heads+h)*r.dimension+c := rfl
    _ < (a*r.heads+h)*r.dimension+r.dimension := Nat.add_lt_add_left hc _
    _ = (a*r.heads+h+1)*r.dimension := by ring
    _ ≤ r.size := Nat.mul_le_mul_right _ (by omega)
theorem offset_injective (r : Request) {a h c a' h' c' : Nat}
    (hh : h<r.heads) (hc : c<r.dimension) (hh' : h'<r.heads) (hc' : c'<r.dimension)
    (he : r.offset a h c=r.offset a' h' c') : a=a' ∧ h=h' ∧ c=c' := by
  have hd : 0<r.dimension := by omega
  have hp : 0<r.heads := by omega
  have cc := congrArg (fun n => n%r.dimension) he
  simp [Request.offset,Nat.add_mod,Nat.mod_eq_of_lt hc,Nat.mod_eq_of_lt hc'] at cc
  have div (k x : Nat) (hx : x<r.dimension) : (k*r.dimension+x)/r.dimension=k := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hd,Nat.div_eq_of_lt hx]; simp
  have ah := congrArg (fun n => n/r.dimension) he
  simp only [Request.offset,div _ _ hc,div _ _ hc'] at ah
  have hh1 := congrArg (fun n => n%r.heads) ah
  simp [Nat.add_mod,Nat.mod_eq_of_lt hh,Nat.mod_eq_of_lt hh'] at hh1
  have divh (k x : Nat) (hx : x<r.heads) : (k*r.heads+x)/r.heads=k := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hx]; simp
  have aa := congrArg (fun n => n/r.heads) ah
  simp only [divh _ _ hh,divh _ _ hh'] at aa
  exact ⟨aa,hh1,cc⟩
theorem coverage (r : Request) (a h c : Nat) (ha : a<r.rows) (hh : h<r.heads) (hc : c<2*r.half) :
    ∃ l∈schedule r,active l ∧ row l=a ∧ head l=h ∧ column l=c := by
  let j := if c<r.half then c else c-r.half
  have hj : j<r.half := by dsimp [j]; split <;> omega
  let i : Lane r := (⟨j,lt_of_lt_of_le hj (block_covers r)⟩,PUnit.unit)
  let l : Location r := (a,h/4,h,decide (r.half≤c),i)
  have hgroup : h/4<r.groups := by unfold Request.groups; omega
  refine ⟨l,(mem_schedule r l).mpr ?_,⟨ha,hh,hj⟩,rfl,rfl,?_⟩
  · dsimp [l,row,group,head]
    exact ⟨ha,hgroup,by omega,lt_min (by omega) hh⟩
  · by_cases hc' : c<r.half
    · have hl : ¬r.half≤c := by omega
      simp [column,l,lane,side,i,j,hc',hl]
    · have hl : r.half≤c := by omega
      simp [column,l,lane,side,i,j,hc',hl]

#axiomsClean coverage
#axiomsClean offset_injective
#axiomsClean offset_lt
end VeriTile.Bench.Repairs.RopeEmbedding
