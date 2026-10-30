import bench.tritonbench_g.sgmv_expand_slice.Proofs.Geometry
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.GridGeometry
open VeriTile.Triton

def localRow (p : Shape) (pid : Nat) (i : TileIndex [32,32]) : Nat :=
  pid / tiles p.columns * 32 + i.1.val
def localColumn (p : Shape) (pid : Nat) (i : TileIndex [32,32]) : Nat :=
  pid % tiles p.columns * 32 + i.2.1.val
def active (p : Shape) (s : Segment) (pid : Nat) (i : TileIndex [32,32]) : Prop :=
  localRow p pid i < s.length ∧ localColumn p pid i < p.columns
instance (p : Shape) (s : Segment) (pid : Nat) : DecidablePred (active p s pid) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))
def row (p : Shape) (s : Segment) (pid : Nat) (i : TileIndex [32,32]) :=
  s.start + localRow p pid i
def column (p : Shape) (pid : Nat) (i : TileIndex [32,32]) :=
  p.slice + localColumn p pid i
def offset (p : Shape) (s : Segment) (pid : Nat) (i : TileIndex [32,32]) :=
  safeRow s (localRow p pid i)*p.outputColumns + safeColumn p (localColumn p pid i)

theorem active_offset (p : Shape) (s : Segment) (pid : Nat) (i : TileIndex [32,32])
    (hi : active p s pid i) :
    offset p s pid i = row p s pid i*p.outputColumns+column p pid i := by
  simp [offset,safeRow,safeColumn,hi.1,hi.2,row,column]

theorem active_offset_injective (p : Shape) (s : Segment) (pid : Nat)
    (capacity : p.slice+p.columns ≤ p.outputColumns)
    (i j : TileIndex [32,32]) (hi : active p s pid i) (hj : active p s pid j)
    (equal : offset p s pid i = offset p s pid j) : i = j := by
  rw [active_offset p s pid i hi,active_offset p s pid j hj] at equal
  have hci : column p pid i < p.outputColumns := by
    have := hi.2
    unfold column
    omega
  have hcj : column p pid j < p.outputColumns := by
    have := hj.2
    unfold column
    omega
  obtain ⟨hr,hc⟩ := linear_injective hci hcj equal
  rcases i with ⟨ir,ic,iu⟩
  rcases j with ⟨jr,jc,ju⟩
  have hir : ir = jr := by apply Fin.ext; simp [row,localRow] at hr; exact hr
  have hic : ic = jc := by apply Fin.ext; simp [column,localColumn] at hc; exact hc
  subst jr
  subst jc
  congr

/-- Every requested coordinate is covered by a specific program and lane.
This concerns the exact host grid formula, independently of runtime delivery. -/
theorem coordinate_covered (p : Shape) (s : Segment) (r c : Nat)
    (hr : r < s.length) (hc : c < p.columns) :
    ∃ pid, pid < tiles s.length*tiles p.columns ∧
      ∃ i : TileIndex [32,32], active p s pid i ∧
        row p s pid i = s.start+r ∧ column p pid i = p.slice+c := by
  let pid := r/32*tiles p.columns+c/32
  let i : TileIndex [32,32] :=
    (⟨r%32,Nat.mod_lt _ (by decide)⟩,⟨c%32,Nat.mod_lt _ (by decide)⟩,PUnit.unit)
  obtain ⟨hrow,hcol,hpid⟩ := one_dimensional_coordinate hr hc
  have hri : localRow p pid i = r := by
    change pid/tiles p.columns*32+r%32=r
    rw [hrow]
    simpa [Nat.mul_comm] using Nat.div_add_mod r 32
  have hci : localColumn p pid i = c := by
    change pid%tiles p.columns*32+c%32=c
    rw [hcol]
    simpa [Nat.mul_comm] using Nat.div_add_mod c 32
  exact ⟨pid,hpid,i,by simpa [active,hri,hci] using And.intro hr hc,
    by simp [row,hri],by simp [column,hci]⟩

/-- Active coordinates from distinct programs cannot overlap inside a batch. -/
theorem program_unique (p : Shape) (s : Segment) (pid qid : Nat)
    (i j : TileIndex [32,32])
    (sameRow : row p s pid i = row p s qid j)
    (sameColumn : column p pid i = column p qid j) : pid = qid := by
  have hr : localRow p pid i = localRow p qid j := by
    simpa only [row,Nat.add_left_cancel_iff] using sameRow
  have hc : localColumn p pid i = localColumn p qid j := by
    simpa only [column,Nat.add_left_cancel_iff] using sameColumn
  obtain ⟨hr',_⟩ := linear_injective i.1.isLt j.1.isLt hr
  obtain ⟨hc',_⟩ := linear_injective i.2.1.isLt j.2.1.isLt hc
  have hpd := Nat.div_add_mod pid (tiles p.columns)
  have hqd := Nat.div_add_mod qid (tiles p.columns)
  rw [hr',hc'] at hpd
  omega

#axiomsClean active_offset_injective
#axiomsClean coordinate_covered
#axiomsClean program_unique
end VeriTile.Bench.Repairs.SgmvExpandSlice.GridGeometry
