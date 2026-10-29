import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.GraphExecution
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeKernel

namespace VeriTile.Bench.Repairs.LayerNormWelfold.KernelNativeResult
open VeriTile.Triton FP.Structural Reuse
noncomputable section

/-- Complete raw typed-source grid against the independent operation graph. -/
theorem kernel_graph (s : State NativeKernel.Term) (x w bias : Nat → NativeKernel.Term)
    (S D : Nat) (Y Mean Rstd : RegionName) (c : Configuration)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd)
    (hx : ∀ row, row < S → ∀ col, col < D → (s.mem "X" (col + D * row)).read .real = x (col + D * row))
    (hw : ∀ col, col < D → (s.mem "W" col).read .real = w col)
    (hb : ∀ col, col < D → (s.mem "Bias" col).read .real = bias col) :
    ∃ final, Source.launch (withEquality NativeKernel.algebra (fun _ _ => Bool.false)) s Y Mean Rstd S D c.block
        (SourceBinding.tree c.tree) = some final ∧
      (∀ row, row < S → final.mem Mean row = .mk .real
        (Native.eval NativeKernel.constructors (KernelGraph.logicalInputs NativeKernel.algebra x w bias D row)
          (Native.meanExpr D c.block c.tree))) ∧
      (∀ row, row < S → final.mem Rstd row = .mk .real
        (Native.eval NativeKernel.constructors (KernelGraph.logicalInputs NativeKernel.algebra x w bias D row)
          (Native.inverseExpr D c.block c.tree))) ∧
      (∀ row, row < S → ∀ col, col < D → final.mem Y (col + D * row) = .mk .bf16
        (Native.eval NativeKernel.constructors (KernelGraph.logicalInputs NativeKernel.algebra x w bias D row)
          (Native.outputExpr D c.block col c.tree))) ∧
      (∀ q off, gridMiss Y Mean Rstd S D q off → final.mem q off = s.mem q off) :=
  KernelGraph.complete NativeKernel.algebra (fun _ _ => Bool.false) NativeKernel.constructors
    NativeKernel.bindings s x w bias S D Y Mean Rstd c sep hx hw hb

/-- Native observation of the complete raw typed-source grid. -/
theorem kernel_native (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (s : State NativeKernel.Term) (xv wv bv : Nat → NativeKernel.Term)
    (S D : Nat) (Y Mean Rstd : RegionName) (c : Configuration)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd)
    (hx : ∀ row, row < S → ∀ col, col < D → (s.mem "X" (col + D * row)).read .real = xv (col + D * row))
    (hw : ∀ col, col < D → (s.mem "W" col).read .real = wv col)
    (hb : ∀ col, col < D → (s.mem "Bias" col).read .real = bv col) :
    let result := NativeKernel.reference I P xv wv bv D c.block
    ∃ final,
      Source.launch (withEquality NativeKernel.algebra (fun _ _ => Bool.false)) s Y Mean Rstd
        S D c.block (SourceBinding.tree c.tree) = some final ∧
      (∀ row, row < S → NativeKernel.observeCell I P (final.mem Mean row) = .mk .real (result row c.tree).mean) ∧
      (∀ row, row < S → NativeKernel.observeCell I P (final.mem Rstd row) = .mk .real (result row c.tree).inverse) ∧
      (∀ row, row < S → ∀ col, col < D → NativeKernel.observeCell I P (final.mem Y (col + D * row)) =
        .mk .bf16 ((result row c.tree).output col)) ∧
      (∀ q off, gridMiss Y Mean Rstd S D q off → final.mem q off = s.mem q off) ∧
      (∀ q off, gridMiss Y Mean Rstd S D q off →
        NativeKernel.observeCell I P (final.mem q off) = NativeKernel.observeCell I P (s.mem q off)) := by
  dsimp only
  obtain ⟨final, run, hm, hi, hy, frame⟩ := kernel_graph s xv wv bv S D Y Mean Rstd c sep hx hw hb
  refine ⟨final, run, ?_, ?_, ?_, frame, ?_⟩
  · intro row hr
    rw [hm row hr]
    rfl
  · intro row hr
    rw [hi row hr]
    rfl
  · intro row hr col hc
    rw [hy row hr col hc]
    rfl
  · intro q off miss
    rw [frame q off miss]

#specNonCircular NativeKernel.reference avoiding [Source.rowKernel, Source.launch, FP.Structural.exec,
  meanValue, inverseValue, rowOutput]
#axiomsClean kernel_graph
#axiomsClean kernel_native
end
end VeriTile.Bench.Repairs.LayerNormWelfold.KernelNativeResult
