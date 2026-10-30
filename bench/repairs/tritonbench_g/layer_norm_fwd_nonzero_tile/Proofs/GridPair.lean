import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.PairIO
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.OrdinaryEquivalence
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.PairSafety
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.NativeImplementation

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.GridPair
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
open OrdinaryEquivalence
noncomputable section
variable {α : Type} [Inhabited α]

theorem invocation_run (A : Algebra α) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (row : Nat) (s : State α) :
    (NativeInterface.invocation A f r stride M N eps p row).run s =
      (candidateInvocation A f r stride M N eps p row).run s :=
  Pair.execution (engine A) f r stride M N eps p (TypedSource.atRow s M row)

/-- Both grids use the same independent row executions and the same disjoint
merge. All final memories, including Boolean mask and statistics cells, agree. -/
def transport_run (A : Algebra α) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (a : NativeInterface.Arrays r M N α) (s t : State α)
    (run : OrdinaryRun ((original A f r stride M N eps p).schedule a)
      ((original A f r stride M N eps p).writes a) s t) :
    OrdinaryRun ((indexCommuted A f r stride M N eps p).schedule a)
      ((indexCommuted A f r stride M N eps p).writes a) s t where
  rows := fun i => run.rows ⟨i.val, by
    simpa only [original, Implementation.original, NativeInterface.io, indexCommuted,
      List.length_ofFn] using i.isLt⟩
  executed := by
    intro i
    have old := run.executed ⟨i.val, by
      simpa only [original, Implementation.original, NativeInterface.io, indexCommuted,
        List.length_ofFn] using i.isLt⟩
    simp only [original, Implementation.original, NativeInterface.io,
      Fin.getElem_fin, List.getElem_ofFn] at old
    change (List.ofFn (fun row : Fin M => candidateInvocation A f r stride M N eps p row.val))[i.val].run s = _
    simp only [List.getElem_ofFn]
    exact (invocation_run A f r stride M N eps p i.val s).symm.trans old
  framed := by
    intro i region off outside
    exact run.framed ⟨i.val, by
      simpa only [original, Implementation.original, NativeInterface.io, indexCommuted,
        List.length_ofFn] using i.isLt⟩ region off outside
  disjoint := by
    intro i j region off hi hj
    exact Fin.ext (congrArg (fun k => k.val) (run.disjoint
      ⟨i.val, by simpa only [original, Implementation.original, NativeInterface.io,
        indexCommuted, List.length_ofFn] using i.isLt⟩
      ⟨j.val, by simpa only [original, Implementation.original, NativeInterface.io,
        indexCommuted, List.length_ofFn] using j.isLt⟩ region off hi hj))
  readback := by
    intro i region off hit
    exact run.readback ⟨i.val, by
      simpa only [original, Implementation.original, NativeInterface.io, indexCommuted,
        List.length_ofFn] using i.isLt⟩ region off hit
  frame := by
    intro region off outside
    apply run.frame
    intro i
    exact outside ⟨i.val, by
      simpa only [original, Implementation.original, NativeInterface.io, indexCommuted,
        List.length_ofFn] using i.isLt⟩

theorem equivalent (A : Algebra α) (cmp : TypedSource.Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ)
    (width : widthCheck N = Bool.true) (sep : Domain.ReadWriteSeparated f r)
    (geometry : NativeGeometry.Geometry f r stride M N) :
    (original A f r stride M N eps p).Equivalent (indexCommuted A f r stride M N eps p) := by
  intro a s hin
  obtain ⟨t, executed, safe, _, frame⟩ :=
    NativeInterface.implements A cmp f r stride M N eps p width sep geometry a trivial s hin
  refine ⟨t, t, executed, ⟨transport_run A f r stride M N eps p a s t executed.some⟩,
    ?_, ?_, frame, frame⟩
  · intro bounds backed
    refine ⟨safe bounds backed, ?_⟩
    intro i
    have old := safe bounds backed ⟨i.val, by
      simpa only [original, Implementation.original, NativeInterface.io, indexCommuted,
        Implementation.toIO, List.length_ofFn] using i.isLt⟩
    simp only [NativeInterface.io, Fin.getElem_fin, List.getElem_ofFn] at old
    change (List.ofFn (fun row : Fin M => candidateInvocation A f r stride M N eps p row.val))[i.val].Safe bounds s
    simp only [List.getElem_ofFn]
    exact Pair.trace_safe (engine A) f r stride M N eps p bounds (TypedSource.atRow s M i.val) old
  · intro o j _
    rfl

#axiomsClean invocation_run
#axiomsClean transport_run
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.GridPair
