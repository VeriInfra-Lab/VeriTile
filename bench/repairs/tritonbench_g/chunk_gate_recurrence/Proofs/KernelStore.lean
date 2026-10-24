import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelMemory

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelStore
open VeriTile.Triton FP.Structural KernelProgram KernelMemory
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
variable {α : Type} [Inhabited α]

theorem mem_congr (r : RegionName) (dtype : FloatDType) (off : Fin 64 → Fin 16 → Nat)
    (value : Lane → α) (u v : State α) (same : u.mem=v.mem) :
    (scatter r dtype off value u).mem=(scatter r dtype off value v).mem := by
  unfold scatter
  generalize TileShape.allIndices [64,16]=indices
  induction indices generalizing u v with
  | nil => exact same
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    simp only [State.write,same]

@[simp] theorem setReg (r : RegionName) (dtype : FloatDType) (off : Fin 64 → Fin 16 → Nat)
    (value : Lane → α) (u : State α) (name : RegName) (d : TileDType) (shape : TileShape)
    (v : Values α d shape) :
    (scatter r dtype off value (u.setReg name d shape v)).mem=(scatter r dtype off value u).mem :=
  mem_congr r dtype off value _ _ rfl

theorem backward_tile_frame (c : Shape) (p : Program c) (t : Fin (c.N-1))
    (value : Lane → α) (u : State α) :
    Frame .backward c p u (scatter "DI" .real (fun e j => address c p t.val e.val j.val) value u) := by
  intro r off outside
  apply State.scatter_frame
  by_cases same : r="DI"
  · subst r
    apply Or.inr
    intro i _ equal
    exact outside (Or.inl ⟨rfl,t,i.1,i.2.1,equal⟩)
  · exact Or.inl same

theorem backward_scalar_frame (c : Shape) (p : Program c) (t : Fin (c.N-1))
    (value : Cell α) (u : State α) :
    Frame .backward c p u (u.write "DG" (gradientAddress c p t.val) value) := by
  intro r off outside
  apply State.write_other
  by_cases same : r="DG"
  · subst r
    right
    intro equal
    exact outside (Or.inr (Or.inl ⟨rfl,t,equal⟩))
  · exact Or.inl same

theorem backward_seed_frame (c : Shape) (p : Program c)
    (value : Lane → α) (u : State α) :
    Frame .backward c p u (scatter "DL" .fp32 (fun e j => dseedAddress c p e.val j.val) value u) := by
  intro r off outside
  apply State.scatter_frame
  by_cases same : r="DL"
  · subst r
    right
    intro i _ equal
    exact outside (Or.inr (Or.inr ⟨rfl,i.1,i.2.1,equal⟩))
  · exact Or.inl same

theorem dseed_unique (c : Shape) (p : Program c) (e k : Fin 64) (j v : Fin 16)
    (same : dseedAddress c p e.val j.val=dseedAddress c p k.val v.val) : e=k ∧ j=v := by
  have eq : address c p 0 e.val j.val=address c p 0 k.val v.val := by
    unfold dseedAddress at same
    unfold address layout launch TritonBenchG.ChunkGateRecurrence.RealProgram.address
    dsimp
    omega
  exact (address_unique c p 0 0 e k j v eq).2

theorem seed_cell (c : Shape) (p : Program c) (value : Lane → α) (u : State α)
    (e : Fin 64) (j : Fin 16) :
    (scatter "DL" .fp32 (fun a b => dseedAddress c p a.val b.val) value u).mem
      "DL" (dseedAddress c p e.val j.val)=.mk .fp32 (value (e,j,PUnit.unit)) := by
  apply State.scatter_readback (shape:=[64,16]) "DL"
    (fun i => dseedAddress c p i.1.val i.2.1.val) (fun i => .mk .fp32 (value i)) u (e,j,PUnit.unit)
  rintro ⟨a,b,⟨⟩⟩ ⟨a',b',⟨⟩⟩ same
  obtain ⟨ha,hb⟩ := dseed_unique c p a a' b b' same
  subst a';subst b';rfl

#axiomsClean backward_tile_frame
#axiomsClean backward_scalar_frame
#axiomsClean backward_seed_frame
#axiomsClean seed_cell
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelStore
