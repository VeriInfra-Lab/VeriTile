import bench.tritonbench_g.attn_fwd_triton.Proofs.GridCoordinates

namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.GridFootprint
open VeriTile.Triton Memory Relocation GridCoordinates
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def writes (A : FlatAlloc) (s : BlockState) (p : Parameters) (BD : Nat) (idx : GridIndex (grid p)) : WriteFootprint :=
  WriteFootprint.activeTileImage A.flat (Output.address (cfg A s p idx) 128 BD)
    (fun i => Output.active (cfg A s p idx) 128 BD i = Bool.true)

def window (A : FlatAlloc) (p : Parameters) (cell : MemCellAddr) : Prop :=
  cell.1 = A.flat ∧ ∃ i : OutputLayout.Coordinate p, OutputLayout.address p i + A.base p.Out = cell.2

theorem address_coordinate (A : FlatAlloc) (s : BlockState) (p : Parameters) (BD : Nat)
    (idx : GridIndex (grid p)) (i : TileIndex [128,BD])
    (hi : Output.active (cfg A s p idx) 128 BD i = Bool.true) :
    Output.address (cfg A s p idx) 128 BD i =
      OutputLayout.address p (OutputLayout.coordinate (cfg A s p idx) 128 BD (program A s p idx) i hi)+A.base p.Out := by
  exact OutputLayout.address_coordinate (cfg A s p idx) 128 BD (program A s p idx) i hi

theorem disjoint (A : FlatAlloc) (s : BlockState) (p : Parameters) (BD : Nat)
    (hn : OutputLayout.NonOverlapping p) (a b : GridIndex (grid p)) (hne : a≠b) :
    WriteFootprint.disjoint (writes A s p BD a) (writes A s p BD b) := by
  rintro cell ⟨_,i,hi,hia⟩ ⟨_,j,hj,hja⟩
  have he := hia.trans hja.symm
  rw [address_coordinate A s p BD a i hi,address_coordinate A s p BD b j hj] at he
  have hc := hn (Nat.add_right_cancel he)
  have hz := congrArg (fun x : OutputLayout.Coordinate p => x.1.val) hc
  have hh := congrArg (fun x : OutputLayout.Coordinate p => x.2.1.val) hc
  have hm := congrArg (fun x : OutputLayout.Coordinate p => x.2.2.1.val) hc
  apply hne
  apply index_eq A s p a b _ hz hh
  change (cfg A s p a).startM*128+i.1.val = (cfg A s p b).startM*128+j.1.val at hm
  have hi' := i.1.isLt
  have hj' := j.1.isLt
  omega

theorem within (A : FlatAlloc) (s : BlockState) (p : Parameters) (BD : Nat)
    (idx : GridIndex (grid p)) (cell : MemCellAddr) (h : writes A s p BD idx cell) : window A p cell := by
  rcases h with ⟨hr,i,hi,he⟩
  refine ⟨hr,OutputLayout.coordinate (cfg A s p idx) 128 BD (program A s p idx) i hi,?_⟩
  exact (address_coordinate A s p BD idx i hi).symm.trans he

def lane (p : Parameters) (BD : Nat) (hD : p.headDim ≤ BD) (i : OutputLayout.Coordinate p) : TileIndex [128,BD] :=
  (⟨i.2.2.1.val%128,by omega⟩,⟨i.2.2.2.val,lt_of_lt_of_le i.2.2.2.isLt hD⟩,PUnit.unit)

theorem lane_row (A : FlatAlloc) (s : BlockState) (p : Parameters) (BD : Nat) (hD : p.headDim ≤ BD)
    (i : OutputLayout.Coordinate p) :
    (cfg A s p (index p i)).startM*128+(lane p BD hD i).1.val = i.2.2.1.val := by
  rw [index_row]
  simp only [lane]
  omega

theorem lane_active (A : FlatAlloc) (s : BlockState) (p : Parameters) (BD : Nat) (hD : p.headDim ≤ BD)
    (i : OutputLayout.Coordinate p) :
    Output.active (cfg A s p (index p i)) 128 BD (lane p BD hD i) = Bool.true := by
  have hm := lane_row A s p BD hD i
  simp only [Output.active,Output.outputConfig,
    VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealEpilogue.active]
  change decide ((cfg A s p (index p i)).startM*128+(lane p BD hD i).1.val < p.context ∧ i.2.2.2.val < p.headDim)=Bool.true
  simp [hm,i.2.2.1.isLt,i.2.2.2.isLt]

theorem lane_address (A : FlatAlloc) (s : BlockState) (p : Parameters) (BD : Nat) (hD : p.headDim ≤ BD)
    (i : OutputLayout.Coordinate p) :
    Output.address (cfg A s p (index p i)) 128 BD (lane p BD hD i) = OutputLayout.address p i+A.base p.Out := by
  rw [address_coordinate A s p BD _ _ (lane_active A s p BD hD i)]
  congr 2
  apply Prod.ext
  · apply Fin.ext
    exact index_batch A s p i
  · apply Prod.ext
    · apply Fin.ext
      exact index_head A s p i
    · apply Prod.ext
      · apply Fin.ext
        exact lane_row A s p BD hD i
      · rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.GridFootprint
