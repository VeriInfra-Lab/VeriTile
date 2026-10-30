import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.KernelReal
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.KernelSource

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.KernelReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPTyped
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (atProgram Outside)
noncomputable section

theorem packed_preserved (initial current : State ℝ) (packed : Packed initial)
    (outside : Outside "C" initial current) : Packed current := by
  intro region hr offset
  rcases hr with rfl|rfl
  · rw [outside "Q" offset (by decide)]
    exact packed "Q" (Or.inl rfl) offset
  · rw [outside "Z" offset (by decide)]
    exact packed "Z" (Or.inr rfl) offset

theorem offset_at (p : Parameters) (s : State ℝ) (pid : Nat) (i : CIndex p) :
    outputOffset p (atProgram s pid) i=Geometry.offset (Grid.geometry p) pid i := by
  simp only [outputOffset,Geometry.offset,Geometry.outRow,Geometry.outColumn,Grid.geometry]
  rw [Nat.mul_comm p.outputRowStride,Nat.mul_comm p.outputColumnStride]
  rfl

theorem single (c : Call) (valid : c.Valid) (i : Fin 17)
    (initial current : State ℝ) (packed : Packed initial)
    (outside : Outside "C" initial current) (pid : Nat) :
    ∃ final, KernelSource.one scalar dot c i pid current=some final ∧
      GridExecution.Post "C" (Geometry.address (Grid.geometry (params c i)))
        (fun x : Grid.Index (params c i) => .mk .fp16
          (Real.mathematical c (RealHost.input c initial) x.1.val x.2.val))
        (Geometry.Owns (Grid.geometry (params c i)) pid) current final := by
  let p := params c i
  let s := atProgram current pid
  have currentPacked : Packed s := packed_preserved initial s packed outside
  have writable : ∀ a b,active p s a → active p s b →
      outputOffset p s a=outputOffset p s b → a=b := by
    intro a b ha hb same
    apply Geometry.offset_injective (Grid.geometry p) (geometry_valid c valid i) pid a b ha hb
    simpa only [s,offset_at] using same
  obtain ⟨final,executed,values,frame⟩ := correctness scalar dot p s writable
  refine ⟨final,executed,?_,?_⟩
  · intro x owns
    obtain ⟨lane,hr,hc⟩ := owns
    change Geometry.outRow (Grid.geometry p) pid lane=x.1.val at hr
    change Geometry.outColumn (Grid.geometry p) pid lane=x.2.val at hc
    have live : active p s lane := Geometry.owns_active (Grid.geometry p) pid x lane hr hc
    have address : outputOffset p s lane=Geometry.address (Grid.geometry p) x := by
      rw [show s=atProgram current pid from rfl,offset_at]
      simp only [Geometry.offset,Geometry.address,hr,hc]
    rw [←address]
    change final.mem p.output (outputOffset p s lane)=_
    rw [values lane live,output_mathematical c valid i s currentPacked lane live]
    change Cell.mk .fp16 (Real.mathematical c (RealHost.input c s)
      (Geometry.outRow (Grid.geometry p) pid lane)
      (Geometry.outColumn (Grid.geometry p) pid lane))=_
    rw [hr,hc,RealHost.inputs_preserved c initial s outside]
  · intro region offset miss
    apply frame region offset
    rcases miss with miss|miss
    · exact Or.inl miss
    · right
      intro lane live
      let x := Geometry.index (Grid.geometry p) pid lane live
      have owns : Geometry.Owns (Grid.geometry p) pid x := ⟨lane,rfl,rfl⟩
      have address : outputOffset p s lane=Geometry.address (Grid.geometry p) x := offset_at p current pid lane
      rw [address]
      exact miss x owns

theorem whole_grid (c : Call) (valid : c.Valid) (i : Fin 17)
    (initial : State ℝ) (packed : Packed initial) :
    ∃ final, KernelSource.run scalar dot c i initial=some final ∧
      GridExecution.Post "C" (Geometry.address (Grid.geometry (params c i)))
        (fun x : Grid.Index (params c i) => .mk .fp16
          (Real.mathematical c (RealHost.input c initial) x.1.val x.2.val))
        (fun _ => True) initial final := by
  obtain ⟨final,executed,values,frame⟩ := GridExecution.compose "C"
    (Geometry.address (Grid.geometry (params c i))) (geometry_valid c valid i).writable
    (fun x : Grid.Index (params c i) => Cell.mk .fp16
      (Real.mathematical c (RealHost.input c initial) x.1.val x.2.val))
    (Geometry.Owns (Grid.geometry (params c i))) (KernelSource.one scalar dot c i)
    initial (KernelSource.count c i)
    (fun pid _ current outside => single c valid i initial current packed outside pid)
    (List.range (KernelSource.count c i)) (fun _ h => List.mem_range.mp h)
    initial (fun _ _ _ => rfl)
  refine ⟨final,executed,?_,?_⟩
  · intro x _
    obtain ⟨pid,bound,owns⟩ := Geometry.covers (Grid.geometry (params c i)) (geometry_valid c valid i) x
    exact values x ⟨pid,List.mem_range.mpr bound,owns⟩
  · intro region offset miss
    exact frame region offset (miss.imp id (fun h x _ => h x trivial))

#axiomsClean whole_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.KernelReal
