import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Proofs.GridProjection
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Proofs.GridSafety

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.IOImplements
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked Execution
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem windows (M : Algebra α) (ops : Operations α) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) (vals : ChannelValues α (IO.inputs g)) (bounds : RegionName → Nat)
    (h : (IO.matmulIO M ops g b).InBounds bounds vals) :
    SourceSafety.Windows g.problem (b.views g) bounds where
  inputA := fun i hi k hk => h.2.1 (0 : Fin 2) (⟨i,hi⟩,⟨k,hk⟩,()) trivial
  inputB := fun k hk j hj => h.2.1 (1 : Fin 2) (⟨k,hk⟩,⟨j,hj⟩,()) trivial
  output := fun i hi j hj => h.2.2.1 (0 : Fin 1) (⟨i,hi⟩,⟨j,hj⟩,()) trivial

theorem outside (M : Algebra α) (ops : Operations α) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) (vals : ChannelValues α (IO.inputs g)) (r : RegionName) (o : Nat)
    (h : (IO.matmulIO M ops g b).Outside vals r o) : ReadProjection.Outside g.problem (b.views g) r o := by
  by_cases hr : r="C"
  · subst r
    right
    intro i hi j hj
    have hh : ("C" : RegionName)≠"C" ∨ o≠i*b.C.rowStride+j*b.C.columnStride :=
      h.1 (0 : Fin 1) (⟨i,hi⟩,⟨j,hj⟩,()) trivial
    exact hh.resolve_left (by simp)
  · exact Or.inl hr

theorem safe (M : Algebra α) (ops : Operations α) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) (bounds : RegionName → Nat)
    (W : SourceSafety.Windows g.problem (b.views g) bounds) (ids : List Nat)
    (valid : ∀ pid∈ids,pid<g.problem.tiles) (s : State α) :
    ScheduleSafe bounds (ids.map (IO.invocation M ops g b)) s := by
  induction ids generalizing s with
  | nil => trivial
  | cons pid tail ih =>
    constructor
    · apply SourceSafety.kernel_safe M (selectedDot ops) g.problem (b.views g) bounds W
      simpa [Invocation.initial,IO.invocation,bindRegisters] using valid pid (by simp)
    · intro t _
      exact ih (fun p hp => valid p (by simp [hp])) t

/-- Composition of proved complete source execution, real access safety,
logical output coverage and exact raw-cell frame. The target premise is an
internal composition fact, discharged by independent formulas in IOResults. -/
theorem implements (M : Algebra α) (ops : Operations α) (law : ScalarLaws M ops)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (L : KernelBuffers.Layout g b)
    (f : ChannelValues α (IO.inputs g) → ChannelValues α (IO.outputs g))
    (formula : ∀ vals initial, (IO.matmulIO M ops g b).InputBinding vals initial →
      ∀ i, ∀ hi : i<g.M, ∀ j, ∀ hj : j<g.N,
        Source.run (sourceArithmetic ops) g.problem (ReadProjection.views (b.views g))
          (ReadProjection.decode (b.views g) initial)
          ((ReadProjection.views (b.views g)).C.address i j)=f vals 0 (⟨i,hi⟩,⟨j,hj⟩,())) :
    (IO.matmulIO M ops g b).Implements f := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    exact safe M ops g b bounds (windows M ops g b vals bounds hb) _
      (fun pid h => (Source.tiles_exact g.problem pid).mp h) initial
  · intro vals initial hin
    obtain ⟨final,ran,projection,tags,frame⟩ := ReadProjection.schedule M ops law g b L
      (Source.tiles g.problem) (fun pid h => (Source.tiles_exact g.problem pid).mp h) initial
    refine ⟨final,ran,?_,?_⟩
    · intro o q _
      fin_cases o
      obtain ⟨pid,hp,lane,ha,hr,hc⟩ := Source.output_covered g.problem q.1.val q.2.1.val q.1.isLt q.2.1.isLt
      have covered : ReadProjection.Covered g.problem (b.views g) (Source.tiles g.problem)
          (q.1.val*b.C.rowStride+q.2.1.val*b.C.columnStride) :=
        ⟨pid,hp,lane,ha,by rw [hr,hc];rfl⟩
      obtain ⟨v,hv⟩ := tags _ (Or.inr covered)
      have eq := congrFun projection ((ReadProjection.views (b.views g)).C.address q.1.val q.2.1.val)
      have address : (ReadProjection.views (b.views g)).C.address q.1.val q.2.1.val=
          (2,(b.views g).C.offset+(q.1.val*b.C.rowStride+q.2.1.val*b.C.columnStride)) := by
        simp [Tensor.address,ReadProjection.views,KernelBuffers.Buffers.views,KernelBuffers.Buffer.view,Nat.add_assoc]
      rw [address] at eq
      rw [ReadProjection.decode_c,hv,Cell.read_same] at eq
      have target := formula vals initial hin q.1.val q.1.isLt q.2.1.val q.2.1.isLt
      rw [address] at target
      have expected := eq.trans target
      change mapCell id (final.mem "C" (q.1.val*b.C.rowStride+q.2.1.val*b.C.columnStride))=
        Cell.mk .fp16 (f vals 0 q)
      rw [hv]
      exact congrArg (Cell.mk .fp16) expected
    · intro r o ho
      exact frame r o (outside M ops g b vals r o ho)

#axiomsClean implements
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.IOImplements
