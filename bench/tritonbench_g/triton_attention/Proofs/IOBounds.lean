import bench.tritonbench_g.triton_attention.Proofs.IOBindings
import bench.tritonbench_g.triton_attention.Proofs.AccessGeometry

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxHeartbeats 2000000

theorem element_capacity (c : Shape) (capacity : Nat)
    (backed : ∀ i : Index c,position c i<capacity) : c.elements≤capacity := by
  by_contra short
  have h : capacity<c.elements := by omega
  have contradiction := backed (index c capacity h)
  rw [position_index] at contradiction
  exact Nat.lt_irrefl _ contradiction

theorem row_capacity (c : Shape) (capacity : Nat)
    (backed : ∀ i : TileIndex [c.groups,c.rows],rowOffset c i.1 i.2.1<capacity) :
    c.groups*c.rows≤capacity := by
  by_contra short
  obtain ⟨head,row,equal⟩ := Erased.row_coord c capacity (by omega)
  have contradiction := backed (head,row,PUnit.unit)
  change rowOffset c head row<capacity at contradiction
  rw [equal] at contradiction
  exact Nat.lt_irrefl _ contradiction

theorem in_bounds {α β γ : Type} (c : Shape) (encode : α→β) (observe : β→α)
    (jobs : List (MaterializedInvocation β γ)) (xs : Inputs α c) (bounds : RegionName→Nat)
    (backed : (layout c encode observe jobs).InBounds bounds xs) : SourceAccess.Bounds c bounds := by
  constructor
  · intro r member
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
    · exact element_capacity c _ (fun i=>backed.2.1 (0 : Fin 5) i trivial)
    · exact element_capacity c _ (fun i=>backed.2.1 (1 : Fin 5) i trivial)
    · exact element_capacity c _ (fun i=>backed.2.1 (2 : Fin 5) i trivial)
    · exact element_capacity c _ (fun i=>backed.2.1 (3 : Fin 5) i trivial)
    · exact element_capacity c _ (fun i=>backed.2.2.1 (0 : Fin 4) i trivial)
    · exact element_capacity c _ (fun i=>backed.2.1 (4 : Fin 5) i trivial)
    · exact element_capacity c _ (fun i=>backed.2.2.1 (2 : Fin 4) i trivial)
    · exact element_capacity c _ (fun i=>backed.2.2.1 (3 : Fin 4) i trivial)
    · exact element_capacity c _ (fun i=>backed.2.2.2 (2 : Fin 4) i trivial)
  · intro r member
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl
    · exact row_capacity c _ (fun i=>backed.2.2.2 (0 : Fin 4) i trivial)
    · exact row_capacity c _ (fun i=>backed.2.2.2 (1 : Fin 4) i trivial)
    · exact row_capacity c _ (fun i=>backed.2.2.2 (3 : Fin 4) i trivial)

/-- The shared exterior frame excludes exactly the original finite output and
scratch windows, including DQ's declared input/output alias. -/
theorem outside {α β γ : Type} (c : Shape) (encode : α→β) (observe : β→α)
    (jobs : List (MaterializedInvocation β γ)) (xs : Inputs α c) (r : RegionName) (o : Nat)
    (exterior : (layout c encode observe jobs).Outside xs r o) : ¬Grid.Written c r o := by
  intro written
  rcases written with ⟨member,inside⟩|⟨member,inside⟩
  · let i := index c o inside
    have equal : position c i=o := position_index c o inside
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl|rfl
    · exact (exterior.1 (0 : Fin 4) i trivial).elim (fun h=>h rfl) (fun h=>h equal.symm)
    · exact (exterior.2 (2 : Fin 4) i trivial).elim (fun h=>h rfl) (fun h=>h equal.symm)
    · exact (exterior.1 (1 : Fin 4) i trivial).elim (fun h=>h rfl) (fun h=>h equal.symm)
    · exact (exterior.1 (2 : Fin 4) i trivial).elim (fun h=>h rfl) (fun h=>h equal.symm)
    · exact (exterior.1 (3 : Fin 4) i trivial).elim (fun h=>h rfl) (fun h=>h equal.symm)
  · obtain ⟨head,row,equal⟩ := Erased.row_coord c o inside
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl
    · exact (exterior.2 (0 : Fin 4) (head,row,PUnit.unit) trivial).elim (fun h=>h rfl) (fun h=>h equal.symm)
    · exact (exterior.2 (1 : Fin 4) (head,row,PUnit.unit) trivial).elim (fun h=>h rfl) (fun h=>h equal.symm)
    · exact (exterior.2 (3 : Fin 4) (head,row,PUnit.unit) trivial).elim (fun h=>h rfl) (fun h=>h equal.symm)

#axiomsClean in_bounds
#axiomsClean outside
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
