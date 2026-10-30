import bench.tritonbench_g.chunk_cumsum_vector.Proofs.Geometry

/-! Geometry shared by the independent Real and FP execution layers. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumVector.Grid
open VeriTile.Triton

@[ext] theorem program_ext {h : Call} (p q : Geometry.Program h)
    (hf : p.feature=q.feature) (hr : p.row=q.row) : p=q := by
  cases p
  cases q
  simp_all

def programs (h : Call) : List (Geometry.Program h) :=
  (List.finRange (Geometry.features h)).flatMap fun feature =>
    (List.finRange (h.B*h.H)).map fun row => ⟨feature.val,row.val,feature.isLt,row.isLt⟩

theorem mem_programs (h : Call) (p : Geometry.Program h) : p∈programs h := by
  apply List.mem_flatMap.mpr
  refine ⟨⟨p.feature,p.featureInside⟩,by simp,?_⟩
  apply List.mem_map.mpr
  exact ⟨⟨p.row,p.rowInside⟩,by simp,by cases p; rfl⟩

def active (h : Call) (p : Geometry.Program h) (t : Nat) (j : Fin BS) : Prop :=
  t<h.T ∧ p.feature*BS+j.val<h.S
def offset (h : Call) (p : Geometry.Program h) (t : Nat) (j : Fin BS) : Nat :=
  h.outputStrides.address h.H p.row t (p.feature*BS+j.val)
def footprint (h : Call) (p : Geometry.Program h) (cell : RegionName × Nat) : Prop :=
  cell.1=h.output ∧ ∃ (t : Nat) (j : Fin BS), active h p t j ∧ cell.2=offset h p t j

theorem footprint_unique (h : Call) (layout : Geometry.OutputLayout h)
    (p q : Geometry.Program h) (t u : Nat) (i j : Fin BS)
    (ht : active h p t i) (hu : active h q u j)
    (eq : offset h p t i=offset h q u j) : p=q ∧ t=u ∧ i=j := by
  obtain ⟨row,time,column⟩ := layout.unique _ _ _ _ _ _ p.rowInside q.rowInside ht.1 hu.1 ht.2 hu.2 eq
  have il := i.isLt
  have jl := j.isLt
  have feat : p.feature=q.feature := by
    have a : (p.feature*BS+i.val)/BS=p.feature := by
      rw [Nat.add_comm,Nat.add_mul_div_right _ _ (by decide : 0<BS),Nat.div_eq_of_lt il,Nat.zero_add]
    have b : (q.feature*BS+j.val)/BS=q.feature := by
      rw [Nat.add_comm,Nat.add_mul_div_right _ _ (by decide : 0<BS),Nat.div_eq_of_lt jl,Nat.zero_add]
    rw [←a,←b,column]
  exact ⟨program_ext p q feat row,time,Fin.ext (by rw [feat] at column; omega)⟩

def owner (h : Call) (row : Fin (h.B*h.H)) (channel : Fin h.S) : Geometry.Program h :=
  ⟨channel.val/BS,row.val,(Geometry.coordinate_cover h.S BS channel.val (by decide) channel.isLt).1,row.isLt⟩
def lane (h : Call) (channel : Fin h.S) : Fin BS := ⟨channel.val%BS,Nat.mod_lt _ (by decide)⟩
theorem owner_coordinate (h : Call) (row : Fin (h.B*h.H)) (channel : Fin h.S) :
    (owner h row channel).feature*BS+(lane h channel).val=channel.val := by
  dsimp [owner,lane]
  rw [Nat.mul_comm]
  exact Nat.div_add_mod channel.val BS

#axiomsClean footprint_unique
#axiomsClean mem_programs
end VeriTile.Bench.Repairs.ChunkCumsumVector.Grid
