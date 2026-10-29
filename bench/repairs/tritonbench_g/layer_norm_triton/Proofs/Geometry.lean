import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Model

namespace VeriTile.Bench.Repairs.LayerNormTriton.Geometry
open VeriTile.Bench.Repairs.FastLayernorm.Storage
noncomputable section

/-- Exactly the accepted normalized trailing axes; zero dimensions are allowed. -/
structure Shape (input weight bias : List Nat) where
  (leading normalized : List Nat)
  nonempty : normalized≠[]
  inputEq : input=leading++normalized
  weightEq : weight=normalized
  biasEq : bias=normalized

def Shape.rows (s : Shape input weight bias) := s.leading.prod
def Shape.columns (s : Shape input weight bias) := s.normalized.prod

theorem shape_product (s : Shape input weight bias) :
    input.prod=s.rows*s.columns ∧ weight.prod=s.columns ∧ bias.prod=s.columns := by
  simp [Shape.rows,Shape.columns,s.inputEq,s.weightEq,s.biasEq,List.prod_append]

def groups (N : Nat) : Nat := if N≤1024 then 256 else if N≤4096 then 128 else if N≤8192 then 96 else 64

def warps (B : Nat) : Nat := min (max (B/256) 1) 8

theorem groups_bounds (N : Nat) : 0<groups N ∧ groups N≤256 := by
  unfold groups
  split <;> try omega
  split <;> try omega
  split <;> omega

theorem warps_bounds (B : Nat) : 1≤warps B ∧ warps B≤8 := by
  unfold warps
  omega

theorem ceil_mem (a G k : Nat) (hg : 0<G) : k<(a+G-1)/G ↔ k*G<a := by
  have hmod := Nat.mod_lt (a+G-1) hg
  have hdiv := Nat.div_add_mod (a+G-1) G
  have hsub : a+G-1+1=a+G := by omega
  constructor
  · intro hk
    have hm := Nat.mul_le_mul_right G hk
    simp only [Nat.succ_eq_add_one] at hm
    have hm2 : k*G+G≤G*((a+G-1)/G) := by simpa [Nat.add_mul,Nat.mul_comm] using hm
    omega
  · intro hk
    by_contra hn
    have hm := Nat.mul_le_mul_right G (show (a+G-1)/G≤k by omega)
    nlinarith

theorem group_mem (M G group row : Nat) (hg : 0<G) (hgroup : group<M) :
    row∈groupRows M G group ↔ ∃ k,row=group+G*k ∧ row<M := by
  rw [groupRows,List.mem_range']
  constructor
  · rintro ⟨k,hk,rfl⟩
    refine ⟨k,rfl,?_⟩
    have := (ceil_mem (M-group) G k hg).mp hk
    rw [Nat.mul_comm] at this
    omega
  · rintro ⟨k,rfl,hr⟩
    refine ⟨k,(ceil_mem (M-group) G k hg).mpr ?_,rfl⟩
    rw [Nat.mul_comm]
    omega

theorem group_residue (M G group row : Nat) (hg : 0<G)
    (hgroup : group<min G M) : row∈groupRows M G group ↔ row<M ∧ row%G=group := by
  rw [group_mem M G group row hg (by omega)]
  constructor
  · rintro ⟨k,rfl,hr⟩
    exact ⟨hr,by simp [Nat.add_mod,Nat.mod_eq_of_lt (show group<G by omega)]⟩
  · rintro ⟨hr,he⟩
    refine ⟨row/G,?_,hr⟩
    have := Nat.mod_add_div row G
    omega

theorem unique_group (M G row : Nat) (hg : 0<G) (hr : row<M) :
    ∃! group,group<min G M ∧ row∈groupRows M G group := by
  have hmod := Nat.mod_lt row hg
  have hle := Nat.mod_le row G
  refine ⟨row%G,⟨by omega,(group_residue M G (row%G) row hg (by omega)).mpr ⟨hr,rfl⟩⟩,?_⟩
  rintro group ⟨hb,hm⟩
  exact ((group_residue M G group row hg hb).mp hm).2.symm

theorem group_nodup (M G group : Nat) (hg : 0<G) : (groupRows M G group).Nodup := by
  exact List.nodup_range' (step:=G) hg

theorem group_increasing (M G group : Nat) (hg : 0<G) :
    (groupRows M G group).Pairwise (·<·) := by
  exact List.pairwise_lt_range' (step:=G) (pos:=hg)

/-- The first failed while-condition is reached after exactly these increments. -/
theorem group_terminates (M G group : Nat) (hg : 0<G) (hgroup : group<M) :
    M≤group+G*((M-group+G-1)/G) := by
  have h := (ceil_mem (M-group) G ((M-group+G-1)/G) hg)
  have hn : ¬ ((M-group+G-1)/G)*G<M-group := by omega
  rw [Nat.mul_comm] at hn
  omega

/-- Bounds come from successful dense allocations, not a caller native-size restriction. -/
theorem grouped_counter_safe (x : Dense) (M N row G : Nat)
    (shape : x.count=M*N) (hn : 0<N) (hr : row<M) (hg : G≤256) : row+G<2^63 := by
  have hc := VeriTile.Bench.Repairs.FastLayernorm.Storage.dense_count_bound x
  have hm := Nat.le_mul_of_pos_right M hn
  norm_num at *
  omega

theorem partial_offset_safe (scratch : Dense) (M N G group col : Nat)
    (shape : scratch.count=min G M*N) (hg : group<min G M) (hc : col<N) :
    group*N+col<scratch.count ∧ group*N+col<2^63 :=
  VeriTile.Bench.Repairs.FastLayernorm.Execution.row_offset_safe scratch (min G M) N group col shape hg hc

theorem source_row_offset_safe (x : Dense) (M N row col : Nat)
    (shape : x.count=M*N) (hr : row<M) (hc : col<N) : row*N+col<x.count ∧ row*N+col<2^63 :=
  VeriTile.Bench.Repairs.FastLayernorm.Execution.row_offset_safe x M N row col shape hr hc

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Geometry
