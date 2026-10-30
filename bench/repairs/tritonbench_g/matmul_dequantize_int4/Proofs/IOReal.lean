import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IOExecution
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IOMath
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IOSafety

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPTyped
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (atProgram Outside)
noncomputable section

theorem schedule_eq (c : RawKernel.Layout) (i : Fin 17) (s : State ℝ) :
    runSchedule (jobs c i) s=KernelSource.run KernelReal.scalar KernelReal.dot c.call i s :=
  IOExecution.schedule_eq _ _ _ _

theorem safe (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17)
    (x : ChannelValues ℝ (IOData.inputs c)) (bounds : RegionName → Nat)
    (hb : (IOData.program ℝ c (jobs c i)).InBounds bounds x) (initial : State ℝ) :
    ScheduleSafe bounds (jobs c i) initial := by
  have w := IOBounds.windows c (jobs c i) x bounds hb
  have every (pid : Nat) (hp : pid<KernelSource.count c.call i) (s : State ℝ) :
      (IOExecution.launch (KernelSource.kernel c.call i)
        (backend KernelReal.scalar KernelReal.dot) pid).Safe bounds s := by
    change TraceSafe (backend KernelReal.scalar KernelReal.dot) (bounds)
      (KernelSource.kernel c.call i) (atProgram s pid)
    exact IOSafety.source_safe _ _ _ bounds (atProgram s pid)
      (IOBounds.steps_positive c valid i) (IOBounds.accesses c valid i bounds w s pid hp)
  have all (pids : List Nat) (listed : ∀ pid∈pids,pid<KernelSource.count c.call i)
      (s : State ℝ) : ScheduleSafe bounds (pids.map
        (IOExecution.launch (KernelSource.kernel c.call i) (backend KernelReal.scalar KernelReal.dot))) s := by
    induction pids generalizing s with
    | nil => trivial
    | cons pid rest ih =>
      exact ⟨every pid (listed pid (by simp)) s,
        fun next _=>ih (fun p hp=>listed p (by simp [hp])) next⟩
  exact all _ (fun _ h=>List.mem_range.mp h) initial

theorem single (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17)
    (x : ChannelValues ℝ (IOData.inputs c)) (packed : IOData.packedInt32 c x)
    (initial current : State ℝ)
    (binding : (IOData.program ℝ c (jobs c i)).InputBinding x initial)
    (outside : Outside "C" initial current) (pid : Nat) :
    ∃ final, KernelSource.one KernelReal.scalar KernelReal.dot c.call i pid current=some final ∧
      GridExecution.Post "C" (Geometry.address (Grid.geometry (params c.call i)))
        (fun j : Grid.Index (params c.call i)=>Cell.mk .fp16
          (Real.mathematical c.call (IOMath.input c x) j.1.val j.2.val))
        (Geometry.Owns (Grid.geometry (params c.call i)) pid) current final := by
  let p := params c.call i
  let s := atProgram current pid
  have hs : (IOData.program ℝ c (jobs c i)).InputBinding x s :=
    IOExecution.binding_preserved c (jobs c i) x initial s binding outside
  have writable : ∀ a b,active p s a → active p s b →
      outputOffset p s a=outputOffset p s b → a=b := by
    intro a b ha hb same
    apply Geometry.offset_injective (Grid.geometry p) (geometry_valid c.call (c.valid valid) i) pid a b ha hb
    simpa only [s,KernelReal.offset_at] using same
  obtain ⟨final,executed,values,frame⟩ := correctness KernelReal.scalar KernelReal.dot p s writable
  refine ⟨final,executed,?_,?_⟩
  · intro j owns
    obtain ⟨lane,hr,hc⟩ := owns
    change Geometry.outRow (Grid.geometry p) pid lane=j.1.val at hr
    change Geometry.outColumn (Grid.geometry p) pid lane=j.2.val at hc
    have live : active p s lane := Geometry.owns_active (Grid.geometry p) pid j lane hr hc
    have address : outputOffset p s lane=Geometry.address (Grid.geometry p) j := by
      rw [show s=atProgram current pid from rfl,KernelReal.offset_at]
      simp only [Geometry.offset,Geometry.address,hr,hc]
    rw [←address]
    change final.mem p.output (outputOffset p s lane)=_
    rw [values lane live,IOMath.output c valid i (jobs c i) x packed s hs lane live]
    change Cell.mk .fp16 (Real.mathematical c.call (IOMath.input c x)
      (Geometry.outRow (Grid.geometry p) pid lane) (Geometry.outColumn (Grid.geometry p) pid lane))=_
    rw [hr,hc]
  · intro region offset miss
    apply frame region offset
    rcases miss with miss|miss
    · exact Or.inl miss
    · right
      intro lane live
      let j := Geometry.index (Grid.geometry p) pid lane live
      have owns : Geometry.Owns (Grid.geometry p) pid j := ⟨lane,rfl,rfl⟩
      have address : outputOffset p s lane=Geometry.address (Grid.geometry p) j :=
        KernelReal.offset_at p current pid lane
      rw [address]
      exact miss j owns

theorem implements (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17) :
    (IOData.program ℝ c (jobs c i)).ImplementsOn (IOData.packedInt32 c) (IOMath.formula c) := by
  intro x packed bounds hb initial binding
  refine ⟨safe c valid i x bounds hb initial,?_⟩
  obtain ⟨final,executed,values,frame⟩ := GridExecution.compose "C"
    (Geometry.address (Grid.geometry (params c.call i))) (geometry_valid c.call (c.valid valid) i).writable
    (fun j : Grid.Index (params c.call i)=>Cell.mk .fp16 (Real.mathematical c.call (IOMath.input c x) j.1.val j.2.val))
    (Geometry.Owns (Grid.geometry (params c.call i))) (KernelSource.one KernelReal.scalar KernelReal.dot c.call i)
    initial (KernelSource.count c.call i)
    (fun pid _ current outside=>single c valid i x packed initial current binding outside pid)
    (List.range (KernelSource.count c.call i)) (fun _ h=>List.mem_range.mp h)
    initial (fun _ _ _=>rfl)
  have post : GridExecution.Post "C" (Geometry.address (Grid.geometry (params c.call i)))
      (fun j : Grid.Index (params c.call i)=>Cell.mk .fp16 (Real.mathematical c.call (IOMath.input c x) j.1.val j.2.val))
      (fun _=>True) initial final := by
    constructor
    · intro j _
      obtain ⟨pid,hp,owns⟩ := Geometry.covers (Grid.geometry (params c.call i))
        (geometry_valid c.call (c.valid valid) i) j
      exact values j ⟨pid,List.mem_range.mpr hp,owns⟩
    · intro r off miss
      exact frame r off (miss.imp id (fun h j _=>h j trivial))
  refine ⟨final,?_,IOExecution.output_binding c (jobs c i) x _ i initial final post⟩
  exact (schedule_eq c i initial).trans executed

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOReal
