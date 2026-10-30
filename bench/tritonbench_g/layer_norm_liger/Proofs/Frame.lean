import bench.tritonbench_g.layer_norm_liger.Proofs.BackwardResult
namespace VeriTile.Bench.Repairs.LayerNormLiger.Backward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section

def Outside {α : Type} (p : Parameters) (s : State α) (r : RegionName) (o : Nat) :=
  (r≠p.dw ∨ ∀ i : TileIndex [p.block], i.1.val<p.columns → o≠s.pids 0*p.dwStride+i.1.val) ∧
  (r≠p.db ∨ ∀ i : TileIndex [p.block], i.1.val<p.columns → o≠s.pids 0*p.dbStride+i.1.val) ∧
  (r≠p.dx ∨ ∀ row, start p s≤row → row<stop p s →
    ∀ i : TileIndex [p.block], i.1.val<p.columns → o≠row*p.dxStride+i.1.val)

theorem reference_frame {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat) (h : Outside p s r o) :
    (reference M R p s).mem r o=s.mem r o := by
  unfold reference
  rw [finish_frame M p _ _ r o h.1 h.2.1]
  exact Logical.loop_frame M R p _ _ _ r o h.2.2

theorem complete_frame {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec (withSum M R) (kernel p) s=some t ∧
      ∀ r o, Outside p s r o → t.mem r o=s.mem r o := by
  obtain ⟨t,ht,hm⟩ := complete_run M R p s
  refine ⟨t,ht,?_⟩
  intro r o h
  rw [hm]
  exact reference_frame M R p s r o h

#axiomsClean reference_frame
#axiomsClean complete_frame
end
end VeriTile.Bench.Repairs.LayerNormLiger.Backward
