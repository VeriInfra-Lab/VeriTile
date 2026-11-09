import VeriTile.Triton.Float.RangeInduction
import bench.tritonbench_g.parallel_attention.Proofs.Support.FPForwardGraph
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
noncomputable section

theorem matrix_index_eta (i : TileIndex [m,n]) : (⟨i.1,i.2.1,()⟩ : TileIndex [m,n])=i := by
  rcases i with ⟨r,e,⟨⟩⟩; rfl


/-- A numerical recurrence may be carried through a source loop without any
associativity or reduction law. The premise proves one actual body execution;
the conclusion follows the original dynamic range indices exactly. -/
theorem range_values {α β : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (body : List ComputeStmt) (update : Nat → β → β)
    (I : Nat → β → State α → Prop) (Done : β → State α → Prop)
    (cur stop stride : Nat) (initial : β) (s : State α)
    (hstep : ∀ n a t, I n a t → n < stop →
      ∃ u, run M body (t.setReg idx .nat [] (fun _ => n)) = some u ∧
        I (n+stride) (update n a) u)
    (hdone : ∀ n a t, I n a t → Done a t)
    (h : I cur initial s) :
    ∃ final, FP.Structural.range M idx cur stop stride body s = some final ∧
      Done (foldRange update cur stop stride initial) final := by
  exact FP.Structural.range_fold M idx body update I Done cur stop stride initial s
    (fun n a => foldRange update n stop stride a)
    (fun n a => by simpa only [dite_eq_ite] using foldRange.eq_def update n stop stride a)
    hstep hdone h

#axiomsClean range_values
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
