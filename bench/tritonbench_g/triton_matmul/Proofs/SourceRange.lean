import VeriTile.Triton.Float.Structural
import bench.tritonbench_g.parallel_attention.Sources.Mathematical_FPRange
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

theorem nat_range_values {α β : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (body : List ComputeStmt) (update : Nat→β→β)
    (I : Nat→β→State α→Prop) (Done : β→State α→Prop)
    (cur stop stride : Nat) (initial : β) (s : State α)
    (hstep : ∀ n a t,I n a t→n<stop→∃ u,
      run M body (t.setReg idx .int [] (fun _=>(n : Int)))=some u ∧ I (n+stride) (update n a) u)
    (hdone : ∀ n a t,I n a t→Done a t) (h : I cur initial s) :
    ∃ final,rangeInt M idx (cur : Int) (stop : Int) stride body s=some final ∧
      Done (foldRange update cur stop stride initial) final := by
  by_cases hz:stride=0
  · exact ⟨s,by simp [rangeInt,hz],by simpa [foldRange,hz] using hdone cur initial s h⟩
  · by_cases hc:cur<stop
    · have hi:(cur : Int)<(stop : Int):=by exact_mod_cast hc
      obtain ⟨next,he,hn⟩:=hstep cur initial s h hc
      obtain ⟨final,hf,hd⟩:=nat_range_values M idx body update I Done
        (cur+stride) stop stride (update cur initial) next hstep hdone hn
      refine ⟨final,?_,?_⟩
      · rw [rangeInt]
        simpa only [hz,hi,↓reduceIte,he,Option.bind_some,Int.natCast_add] using hf
      · rw [foldRange]
        simpa only [hz,hc,↓reduceDIte] using hd
    · have hi:¬(cur : Int)<(stop : Int):=by exact_mod_cast hc
      exact ⟨s,by simp [rangeInt,hz,hi],by simpa [foldRange,hz,hc] using hdone cur initial s h⟩
termination_by stop-cur
decreasing_by omega

#axiomsClean nat_range_values
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
