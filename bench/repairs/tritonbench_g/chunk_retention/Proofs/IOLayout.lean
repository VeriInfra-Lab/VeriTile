import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.IO

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
noncomputable section

def observation (c : Shape) (port : Fin (outputs c).length) :
    TileIndex (outputs c)[port].shape → Observation c := match port with
  | ⟨0,_⟩ => fun j => .history j.1.val j.2
  | ⟨1,_⟩ => fun j => .finalState j
  | ⟨2,_⟩ => fun j => .output j
  | ⟨3,_⟩ => fun j => .dhistory j.1.val j.2
  | ⟨4,_⟩ => fun j => .query j
  | ⟨5,_⟩ => fun j => .key j
  | ⟨6,_⟩ => fun j => .valuePart j.1.val j.2

theorem observation_region (c : Shape) (port : Fin (outputs c).length)
    (j : TileIndex (outputs c)[port].shape) :
    (observation c port j).region=(outputs c)[port].buf := by
  rcases port with ⟨port,h⟩
  have hp : port<7 := h
  interval_cases port <;> rfl

theorem observation_offset {α : Type} (c : Shape) (s : State α)
    (port : Fin (outputs c).length) (j : TileIndex (outputs c)[port].shape) :
    (observation c port j).offset s=outputOffset c s.pids s.numPids port j := by
  rcases port with ⟨port,h⟩
  have hp : port<7 := h
  interval_cases port <;> rfl

theorem observation_active {α : Type} (a : Call) (s : State α)
    (port : Fin (outputs a.shape).length) (j : TileIndex (outputs a.shape)[port].shape) :
    Active a s (observation a.shape port j) ↔ outputActive a s.pids port j := by
  rcases port with ⟨port,h⟩
  have hp : port<7 := h
  interval_cases port
  all_goals
    simp only [observation,Active,outputActive,ForwardHSource.active,ForwardHSource.feature,
      ForwardHSource.column,ForwardOSource.active,ForwardOSource.row,ForwardOSource.column,
      BackwardHSource.active,BackwardHSource.feature,BackwardHSource.column,
      BackwardQKVSource.keyActive,BackwardQKVSource.valueActive,BackwardQKVSource.row,
      BackwardQKVSource.feature,BackwardQKVSource.column]
  all_goals simp only [j.1.isLt,true_and,and_true]

/-- Every previously stated active output is present in a public output
channel; the finite shape does not drop any history or partial-gradient store. -/
theorem active_observation_covered {α : Type} (a : Call) (s : State α)
    (o : Observation a.shape) (h : Active a s o) :
    ∃ port,∃ j : TileIndex (outputs a.shape)[port].shape,
      observation a.shape port j=o ∧ outputActive a s.pids port j := by
  cases o with
  | history n i =>
    exact ⟨0,(⟨n,h.2.1⟩,i),rfl,h.1,h.2.2⟩
  | finalState i => exact ⟨1,i,rfl,h⟩
  | output i => exact ⟨2,i,rfl,h⟩
  | dhistory n i => exact ⟨3,(⟨n,h.2.1⟩,i),rfl,h.1,h.2.2⟩
  | query i => exact ⟨4,i,rfl,h⟩
  | key i => exact ⟨5,i,rfl,h⟩
  | valuePart f i => exact ⟨6,(⟨f,h.2.1⟩,i),rfl,h.1,h.2.2⟩

theorem outside_iff {α β : Type} (M : Algebra β) (encode : α→β) (observe : β→α)
    (a : Call) (s : State β) (xs : Inputs α a.shape) (r : RegionName) (offset : Nat) :
    (io M encode observe a s.pids s.numPids).Outside xs r offset ↔ ¬Footprint a s r offset := by
  constructor
  · intro outside ⟨o,active,hr,ho⟩
    obtain ⟨port,j,rfl,hj⟩ := active_observation_covered a s o active
    have h := outside.1 port j hj
    change r≠(outputs a.shape)[port].buf ∨ offset≠outputOffset a.shape s.pids s.numPids port j at h
    exact h.elim (fun h=>h (hr.trans (observation_region a.shape port j)))
      (fun h=>h (ho.trans (observation_offset a.shape s port j)))
  · intro outside
    constructor
    · intro port j hj
      change r≠(outputs a.shape)[port].buf ∨ offset≠outputOffset a.shape s.pids s.numPids port j
      by_contra h
      simp only [not_or,not_not] at h
      exact outside ⟨observation a.shape port j,(observation_active a s port j).mpr hj,
        h.1.trans (observation_region a.shape port j).symm,
        h.2.trans (observation_offset a.shape s port j).symm⟩
    · intro port
      exact port.elim0

theorem legal_iff {α : Type} (a : Call) (s : State α) :
    Legal a s.pids s.numPids ↔ KernelPacket.Legal a s := Iff.rfl

#axiomsClean observation_offset
#axiomsClean observation_active
#axiomsClean active_observation_covered
#axiomsClean outside_iff
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
