import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.NativeIO

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.NativeInterface
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000
variable {α : Type} [Inhabited α]

theorem invocation_initial (A : Algebra α) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (row : Nat) (s : State α) :
    (invocation A f r stride M N eps p row).initial s=TypedSource.atRow s M row := rfl

/-- Direct transport of the original same-initial-state executions and their
source-derived disjoint merge; no serial replacement or rerun is introduced. -/
def ordinary_of_original (A : Algebra α) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (a : Arrays r M N α) (s t : State α)
    (run : TypedSource.GridRun A f r stride M N eps p (tile N) (tile_arange N) s t) :
    OrdinaryRun ((io A f r stride M N eps p).schedule a)
      ((io A f r stride M N eps p).writes a) s t where
  rows:=fun i=>run.rows ⟨i.val,by simpa only [io,List.length_ofFn] using i.isLt⟩
  executed:=by
    intro i
    change (List.ofFn (fun row : Fin M => invocation A f r stride M N eps p row.val))[i.val].run s = _
    simp only [List.getElem_ofFn]
    simpa only [Invocation.run,invocation_initial,invocation,source_exact]
      using run.executed ⟨i.val,by simpa only [io,List.length_ofFn] using i.isLt⟩
  framed:=by
    intro i region off outside
    exact run.framed ⟨i.val,by simpa only [io,List.length_ofFn] using i.isLt⟩ region off outside
  disjoint:=by
    intro i j region off hi hj
    exact Fin.ext (congrArg (fun k : Fin M => k.val) (run.disjoint
      ⟨i.val,by simpa only [io,List.length_ofFn] using i.isLt⟩
      ⟨j.val,by simpa only [io,List.length_ofFn] using j.isLt⟩ region off hi hj))
  readback:=by
    intro i region off hit
    exact run.readback ⟨i.val,by simpa only [io,List.length_ofFn] using i.isLt⟩ region off hit
  frame:=by
    intro region off outside
    apply run.frame
    intro i
    exact outside ⟨i.val,by simpa only [io,List.length_ofFn] using i.isLt⟩

#axiomsClean ordinary_of_original
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.NativeInterface
