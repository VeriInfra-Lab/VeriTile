import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.IO
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.RealInterface
namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.LogicalReader
open VeriTile Triton FP.Structural AttentionFwdTriton1.Whole001
noncomputable section
theorem mainHead_connection (c : Shape) (i : Fin c.mainSize) :
    (LogicalIO.mainCoordinates c i).1.val = KernelIO.mainHead c i.val := by
  rfl
theorem mainRow_connection (c : Shape) (i : Fin c.mainSize) :
    (LogicalIO.mainCoordinates c i).2.1.val = KernelIO.mainRow c i.val := by
  rfl
theorem mainColumn_connection (c : Shape) (i : Fin c.mainSize) :
    (LogicalIO.mainCoordinates c i).2.2.val = KernelIO.mainColumn c i.val := by
  change i.val % (c.T*c.D) % c.D = i.val % c.D
  exact Nat.mod_mul_left_mod _ _ _
theorem historyHead_connection (c : Shape) (i : Fin c.historySize) :
    (LogicalIO.historyCoordinates c i).1.val = KernelIO.historyHead c i.val := by
  change i.val / (c.chunks*(c.D*c.D)) = i.val / (c.chunks*c.D*c.D)
  rw [Nat.mul_assoc]
theorem historyChunk_connection (c : Shape) (i : Fin c.historySize) :
    (LogicalIO.historyCoordinates c i).2.1.val = KernelIO.historyChunk c i.val := by
  change i.val % (c.chunks*(c.D*c.D)) / (c.D*c.D) = i.val % (c.chunks*c.D*c.D) / (c.D*c.D)
  rw [Nat.mul_assoc]
theorem historyRow_connection (c : Shape) (i : Fin c.historySize) :
    (LogicalIO.historyCoordinates c i).2.2.1.val = KernelIO.historyRow c i.val := by
  change (i.val % (c.chunks*(c.D*c.D))) % (c.D*c.D) / c.D =
    (i.val % (c.chunks*c.D*c.D)) / c.D % c.D
  rw [Nat.mod_mul_right_div_self,Nat.mul_assoc]
theorem historyColumn_connection (c : Shape) (i : Fin c.historySize) :
    (LogicalIO.historyCoordinates c i).2.2.2.val = KernelIO.historyColumn c i.val := by
  change (i.val % (c.chunks*(c.D*c.D))) % (c.D*c.D) % c.D = i.val % c.D
  rw [Nat.mod_mul_left_mod]
  rw [← Nat.mul_assoc,Nat.mod_mul_left_mod]

/-- The entire source/safety/output/frame judgment is transported exactly. -/
theorem statement_iff (c : Shape) (storeFlag ifcond : Bool)
    (f : LogicalIO.Inputs c → LogicalIO.Outputs c)
    (connection : (LogicalIO.io c storeFlag ifcond).lift f = RealIO.formula c) :
    (LogicalIO.io c storeFlag ifcond).Implements f ↔
      (RealIO.io c storeFlag ifcond).Implements (RealIO.formula c) := by
  change (RealIO.io c storeFlag ifcond).Implements ((LogicalIO.io c storeFlag ifcond).lift f) ↔ _
  rw [connection]

theorem formula_connection (c : Shape) (storeFlag ifcond : Bool)
    (f : LogicalIO.Inputs c → LogicalIO.Outputs c)
    (output : ∀ raw h row col,
      (f (LogicalIO.inputsEquiv c raw)).output h row col =
        MathTarget.output c (KernelIO.logicalInputs c raw) (KernelIO.realScale c)
          h.val row.val col.val)
    (history : ∀ raw h chunk row col,
      (f (LogicalIO.inputsEquiv c raw)).history h chunk row col =
        MathTarget.history c (KernelIO.logicalInputs c raw)
          h.val chunk.val row.val col.val) :
    (LogicalIO.io c storeFlag ifcond).lift f = RealIO.formula c := by
  funext raw o j
  fin_cases o
  · rcases j with ⟨j,⟨⟩⟩
    change (f (LogicalIO.inputsEquiv c raw)).output
      (LogicalIO.mainCoordinates c j).1 (LogicalIO.mainCoordinates c j).2.1
      (LogicalIO.mainCoordinates c j).2.2 = _
    rw [output,mainHead_connection,mainRow_connection,mainColumn_connection]
    rfl
  · rcases j with ⟨j,⟨⟩⟩
    change (f (LogicalIO.inputsEquiv c raw)).history
      (LogicalIO.historyCoordinates c j).1 (LogicalIO.historyCoordinates c j).2.1
      (LogicalIO.historyCoordinates c j).2.2.1 (LogicalIO.historyCoordinates c j).2.2.2 = _
    rw [history,historyHead_connection,historyChunk_connection,historyRow_connection,
      historyColumn_connection]
    rfl

theorem correct (c : Shape) (storeFlag ifcond : Bool)
    (f : LogicalIO.Inputs c → LogicalIO.Outputs c)
    (output : ∀ raw h row col,
      (f (LogicalIO.inputsEquiv c raw)).output h row col =
        MathTarget.output c (KernelIO.logicalInputs c raw) (KernelIO.realScale c)
          h.val row.val col.val)
    (history : ∀ raw h chunk row col,
      (f (LogicalIO.inputsEquiv c raw)).history h chunk row col =
        MathTarget.history c (KernelIO.logicalInputs c raw)
          h.val chunk.val row.val col.val) :
    (LogicalIO.io c storeFlag ifcond).Implements f :=
  (statement_iff c storeFlag ifcond f
    (formula_connection c storeFlag ifcond f output history)).mpr
      (RealIO.implements c storeFlag ifcond)

#axiomsClean statement_iff
#axiomsClean formula_connection
#axiomsClean correct
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.LogicalReader
