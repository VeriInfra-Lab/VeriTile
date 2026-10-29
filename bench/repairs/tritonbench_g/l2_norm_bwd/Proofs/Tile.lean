import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Statement

namespace VeriTile.Bench.Repairs.L2NormBwd.Tile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole (writeCell scatter_mem fold_written fold_frame)
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def memory {α : Type} [Inhabited α] (M : Algebra α) (layout : Strides) (N B row : Nat)
    (eps : α) (s : State α) : RegionName → Nat → Cell α :=
  (TileShape.allIndices [B]).foldl (fun mem i =>
    if i.1.val<N then writeCell mem "DX" (row*layout.dxRow+i.1.val*layout.dxCol)
      (.mk .real (scalar M N B eps (input s "X" row layout.xRow layout.xCol)
        (input s "DY" row layout.dyRow layout.dyCol) i.1.val))
    else mem) s.mem

theorem sourceRun_mem {α : Type} [Inhabited α] (M : Algebra α) (layout : Strides) (N B row : Nat)
    (eps : α) (s : State α) : (sourceRun M layout N B row eps s).val.mem=memory M layout N B row eps s := by
  simp only [sourceRun,state14,scatter_mem,memory,state13,state12,state11,state10,
    state9,state8,state7,state6,state5,state4,state3,state2,state1,state0,rowState,State.setReg_mem]
  congr 1
  funext mem i
  by_cases hi : i.1.val<N
  · simp only [hi,if_true]
    rw [tileScalar_active M N B eps _ _ i.1 hi]
    rfl
  · simp only [hi,if_false]

theorem readback {α : Type} [Inhabited α] (M : Algebra α) (layout : Strides) (N B row : Nat)
    (eps : α) (s : State α) (target : TileIndex [B]) (active : target.1.val<N)
    (injective : ∀ i j,i<N → j<N → i*layout.dxCol=j*layout.dxCol → i=j) :
    (sourceRun M layout N B row eps s).val.mem "DX" (row*layout.dxRow+target.1.val*layout.dxCol)=
      .mk .real (scalar M N B eps (input s "X" row layout.xRow layout.xCol)
        (input s "DY" row layout.dyRow layout.dyCol) target.1.val) := by
  rw [sourceRun_mem]
  have unique : ∀ i : TileIndex [B],i.1.val<N →
      row*layout.dxRow+i.1.val*layout.dxCol=row*layout.dxRow+target.1.val*layout.dxCol → i=target := by
    intro i hi h
    apply Prod.ext
    · apply Fin.ext
      exact injective _ _ hi active (Nat.add_left_cancel h)
    · exact Subsingleton.elim _ _
  obtain ⟨before,after,hall⟩ := List.append_of_mem (TileShape.mem_allIndices [B] target)
  unfold memory
  rw [hall,List.foldl_append,List.foldl_cons]
  simp only [active,if_true]
  apply fold_written _ _ _ _ target unique
  simp [writeCell]

theorem frame {α : Type} [Inhabited α] (M : Algebra α) (layout : Strides) (N B row : Nat)
    (eps : α) (s : State α) (region : RegionName) (address : Nat)
    (outside : region≠"DX" ∨ ∀ i : TileIndex [B],i.1.val<N → address≠row*layout.dxRow+i.1.val*layout.dxCol) :
    (sourceRun M layout N B row eps s).val.mem region address=s.mem region address := by
  rw [sourceRun_mem]
  exact fold_frame _ _ _ _ _ _ _ _ outside

#axiomsClean sourceRun
#axiomsClean sourceRun_mem
#axiomsClean readback
#axiomsClean frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Tile
