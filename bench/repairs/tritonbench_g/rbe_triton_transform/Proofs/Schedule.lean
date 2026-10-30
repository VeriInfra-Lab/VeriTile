import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.Spec

namespace VeriTile.Bench.Repairs.RbeTritonTransform
open VeriTile.Triton
set_option maxHeartbeats 1000000

def parts (r : Request) := (r.columns+1023)/1024
def rowParts (r : Request) := (r.rows+1)/2
def grid (r : Request) := rowParts r * parts r
abbrev Lane := TileIndex [2,1024/2]
abbrev Location := Nat × Nat × Lane × Bool
def batch (l : Location) := l.1
def program (l : Location) := l.2.1
def lane (l : Location) := l.2.2.1
def imaginary (l : Location) := l.2.2.2
def row (r : Request) (l : Location) := program l / parts r * 2 + (lane l).1.val
def evenColumn (r : Request) (l : Location) := program l % parts r * 1024 + (lane l).2.1.val*2
def column (r : Request) (l : Location) := evenColumn r l + if imaginary l then 1 else 0
def active (r : Request) (l : Location) : Prop := batch l < r.batches ∧ row r l < r.rows ∧ column r l < r.columns
instance (r : Request) (l : Location) : Decidable (active r l) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def schedule (r : Request) : List Location :=
  (List.range r.batches).flatMap fun b => (List.range (grid r)).flatMap fun p =>
    [Bool.false,Bool.true].flatMap fun im => (TileShape.allIndices [2,1024/2]).map fun i => (b,p,i,im)

theorem parts_cover (r : Request) : r.columns ≤ parts r*1024 := by
  have hm := Nat.mod_lt (r.columns+1023) (by decide : 0<1024)
  have he := Nat.mod_add_div (r.columns+1023) 1024
  unfold parts
  omega

theorem row_parts_cover (r : Request) : r.rows ≤ rowParts r*2 := by
  have hm := Nat.mod_lt (r.rows+1) (by decide : 0<2)
  have he := Nat.mod_add_div (r.rows+1) 2
  unfold rowParts
  omega

theorem mem_schedule (r : Request) (l : Location) :
    l ∈ schedule r ↔ batch l < r.batches ∧ program l < grid r := by
  simp only [schedule,List.mem_flatMap,List.mem_range,List.mem_map]
  constructor
  · rintro ⟨b,hb,p,hp,im,him,i,hi,he⟩
    subst l
    exact ⟨hb,hp⟩
  · intro h
    refine ⟨batch l,h.1,program l,h.2,imaginary l,?_,lane l,?_,?_⟩
    · cases imaginary l <;> simp
    · exact TileShape.mem_allIndices _ _
    · rfl

theorem even_column (r : Request) (l : Location) : evenColumn r l % 2 = 0 := by
  simp [evenColumn,Nat.add_mod,Nat.mul_mod]

theorem column_pair (r : Request) (l : Location) :
    column r l / 2 * 2 = evenColumn r l ∧ decide (column r l % 2 = 1) = imaginary l := by
  have hn := even_column r l
  have he := Nat.mod_add_div (evenColumn r l) 2
  unfold column
  cases h : imaginary l <;> simp only [h, Bool.false_eq_true, ↓reduceIte]
  · constructor
    · omega
    · simp [hn]
  · constructor
    · omega
    · have : (evenColumn r l+1)%2=1 := by omega
      simp [this]

theorem offset_lt (r : Request) (b m c : Nat) (hb : b<r.batches) (hm : m<r.rows) (hc : c<r.columns) :
    r.offset b m c < r.size := by
  have h1 : b*r.rows+m < r.batches*r.rows := by
    have := Nat.mul_le_mul_right r.rows (show b+1 ≤ r.batches by omega)
    rw [Nat.add_mul,Nat.one_mul] at this
    omega
  calc
    r.offset b m c = (b*r.rows+m)*r.columns+c := rfl
    _ < (b*r.rows+m)*r.columns+r.columns := Nat.add_lt_add_left hc _
    _ = (b*r.rows+m+1)*r.columns := by ring
    _ ≤ r.size := Nat.mul_le_mul_right _ (by omega)

theorem offset_injective (r : Request) {b m c b' m' c' : Nat}
    (hm : m<r.rows) (hc : c<r.columns) (hm' : m'<r.rows) (hc' : c'<r.columns)
    (h : r.offset b m c = r.offset b' m' c') : b=b' ∧ m=m' ∧ c=c' := by
  have hk : 0<r.columns := by omega
  have hr : 0<r.rows := by omega
  have hh := congrArg (fun n => n%r.columns) h
  simp [Request.offset,Nat.add_mod,Nat.mod_eq_of_lt hc,Nat.mod_eq_of_lt hc'] at hh
  have divc (a z : Nat) (hz : z<r.columns) : (a*r.columns+z)/r.columns=a := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hk,Nat.div_eq_of_lt hz]
    simp
  have ht := congrArg (fun n => n/r.columns) h
  simp only [Request.offset,divc _ _ hc,divc _ _ hc'] at ht
  have hrow := congrArg (fun n => n%r.rows) ht
  simp [Nat.add_mod,Nat.mod_eq_of_lt hm,Nat.mod_eq_of_lt hm'] at hrow
  have divr (a z : Nat) (hz : z<r.rows) : (a*r.rows+z)/r.rows=a := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hr,Nat.div_eq_of_lt hz]
    simp
  have hbatch := congrArg (fun n => n/r.rows) ht
  simp only [divr _ _ hm,divr _ _ hm'] at hbatch
  exact ⟨hbatch,hrow,hh⟩

theorem coverage (r : Request) (b m c : Nat) (hb : b<r.batches) (hm : m<r.rows) (hc : c<r.columns) :
    ∃ l ∈ schedule r, active r l ∧ batch l=b ∧ row r l=m ∧ column r l=c := by
  have hp : 0<parts r := by have := parts_cover r; omega
  have hcp : c/1024 < parts r := by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0<1024)).mpr
    have := parts_cover r
    omega
  have hmp : m/2 < rowParts r := by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0<2)).mpr
    have := row_parts_cover r
    omega
  let i : Lane := (⟨m%2,Nat.mod_lt _ (by decide)⟩,⟨(c%1024)/2,by omega⟩,PUnit.unit)
  let l : Location := (b,m/2*parts r+c/1024,i,decide (c%2=1))
  have hr : row r l=m := by
    dsimp [row,program,l,lane,i]
    rw [Nat.add_comm (m/2*parts r) (c/1024),Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hcp]
    have := Nat.mod_add_div m 2
    omega
  have hn : evenColumn r l = c/1024*1024+(c%1024)/2*2 := by
    dsimp [evenColumn,program,l,lane,i]
    simp [Nat.add_mod,Nat.mod_eq_of_lt hcp]
  have hcol : column r l=c := by
    unfold column
    rw [hn]
    dsimp [imaginary,l]
    have h1 := Nat.mod_add_div c 1024
    have h2 := Nat.mod_add_div (c%1024) 2
    have h3 := Nat.mod_lt c (by decide : 0<2)
    have h4 : c%1024%2=c%2 := by omega
    split_ifs with hx
    · have hx : c%2=1 := by simpa using hx
      omega
    · have hx : c%2≠1 := by simpa using hx
      omega
  refine ⟨l,(mem_schedule r l).mpr ⟨hb,?_⟩,?_,rfl,hr,hcol⟩
  · dsimp [program,l,grid]
    have hx := Nat.mul_le_mul_right (parts r) (show m/2+1 ≤ rowParts r by omega)
    simp only [Nat.add_mul,Nat.one_mul] at hx
    omega
  · exact ⟨hb,hr ▸ hm,hcol ▸ hc⟩

#axiomsClean coverage
#axiomsClean offset_injective
#axiomsClean offset_lt
#axiomsClean column_pair
end VeriTile.Bench.Repairs.RbeTritonTransform
