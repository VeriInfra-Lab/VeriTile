import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.Runtime
import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.RawResults
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform Ordered
noncomputable section

/-- Complete selected source grid, independent rotation and exact cell frame. -/
theorem kernel_correct (r : Request) (X Y : RegionName) (different : X≠Y)
    (before : State ℝ) (qs : List Runtime.Program) (all : qs.Perm (Runtime.programs r)) :
    VeriTile.Spec.Real (∃ final,Runtime.run ideal r X Y qs before=some final ∧
      (∀ b m c,b<r.batches → m<r.rows → c<r.columns →
        (final.mem Y (r.offset b m c)).read .fp32=
          Spec.result r (fun o => (before.mem X o).read .fp32) b m c) ∧
      (∀ region o,region≠Y ∨ r.size≤o → final.mem region o=before.mem region o)) := by
  obtain ⟨final,ran,values,frame⟩ := Runtime.whole ideal r X Y qs all before different
  exact ⟨final,ran,fun b m c hb hm hc => (values b m c hb hm hc).trans (ideal_result r _ b m c),frame⟩


/-- Complete selected source over raw tagged FP32 cells with the original
fixed primitive/contraction profile, native output words and exact cell frame. -/
theorem kernel_native_equivalence (p : Primitives) (r : Request)
    (X Y : RegionName) (different : X≠Y) (before : State Tagged)
    (qs : List Runtime.Program) (all : qs.Perm (Runtime.programs r)) :
    ∃ final,Runtime.run (operations p) r X Y qs before=some final ∧
      (∀ b m c,b<r.batches → m<r.rows → c<r.columns →
        ((final.mem Y (r.offset b m c)).read .fp32).word=
          Ordered.result p r (fun o => ((before.mem X o).read .fp32).word) b m c) ∧
      (∀ region o,region≠Y ∨ r.size≤o → final.mem region o=before.mem region o) := by
  obtain ⟨final,ran,values,frame⟩ := Runtime.whole (operations p) r X Y qs all before different
  refine ⟨final,ran,?_,frame⟩
  intro b m c hb hm hc
  rw [values b m c hb hm hc,source_profile]


#axiomsClean kernel_correct
#axiomsClean kernel_native_equivalence
end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.RawResults
