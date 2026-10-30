import bench.tritonbench_g.layer_norm_triton.IO
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceSafety
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceForwardResult

namespace VeriTile.Bench.Repairs.LayerNormTriton.ForwardIO
open VeriTile.Triton FP.Structural SourceModel Interface
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

/-- The literal ordered source graph depends only on the active logical row. -/
theorem graph_congr (m : Model α) (N B : Nat) (eps : ℝ)
    (x y w v bias alt : Nat → α)
    (hx : ∀ j,j<N → x j=y j) (hw : ∀ j,j<N → w j=v j)
    (hb : ∀ j,j<N → bias j=alt j) :
    SourceForwardGraph.mean m N B x=SourceForwardGraph.mean m N B y ∧
    SourceForwardGraph.inverse m N B eps x=SourceForwardGraph.inverse m N B eps y ∧
    ∀ j,j<N → SourceForwardGraph.output m N B eps x w bias j=
      SourceForwardGraph.output m N B eps y v alt j := by
  have loads : SourceForwardGraph.xload m N x=SourceForwardGraph.xload m N y := by
    funext j
    by_cases hj : j<N
    · simp [SourceForwardGraph.xload,hj,hx j hj]
    · simp [SourceForwardGraph.xload,hj]
  have means : SourceForwardGraph.mean m N B x=SourceForwardGraph.mean m N B y := by
    simp only [SourceForwardGraph.mean,loads]
  have centered : SourceForwardGraph.centered m N B x=SourceForwardGraph.centered m N B y := by
    funext j
    simp only [SourceForwardGraph.centered,congrFun loads j,means]
  have inverses : SourceForwardGraph.inverse m N B eps x=SourceForwardGraph.inverse m N B eps y := by
    simp only [SourceForwardGraph.inverse,SourceForwardGraph.variance,centered]
  refine ⟨means,inverses,?_⟩
  intro j hj
  simp only [SourceForwardGraph.output,loads,means,inverses,hw j hj,hb j hj]

def expected (m : Model α) (stride N B row : Nat) (eps : ℝ)
    (observe : RegionName → Nat → α → α) (vals : Forward.Inputs α N) : Forward.Outputs α N :=
  let x := Forward.value vals 0
  let w := Forward.value vals 1
  let bias := Forward.value vals 2
  Forward.result
    (fun j => observe "Y" (row*stride+j.val) (SourceForwardGraph.output m N B eps x w bias j.val))
    (observe "Mean" row (SourceForwardGraph.mean m N B x))
    (observe "Rstd" row (SourceForwardGraph.inverse m N B eps x))

theorem implements (m : Model α) (stride N B row : Nat) (eps : ℝ)
    (observe : RegionName → Nat → α → α) (positive : 0<N) (covers : N≤B) :
    (Forward.io m stride N B row eps observe).Implements (expected m stride N B row eps observe) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds boundsOK initial _
    change ScheduleSafe bounds [job m (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) row] initial
    rw [scheduleSafe_singleton]
    apply SourceSafety.forward_safe m stride N B eps positive covers
    · intro j hj
      exact boundsOK.2.1 (0 : Fin 3) (⟨j,hj⟩,()) trivial
    · intro j hj
      exact boundsOK.2.1 (1 : Fin 3) (⟨j,hj⟩,()) trivial
    · intro j hj
      exact boundsOK.2.1 (2 : Fin 3) (⟨j,hj⟩,()) trivial
    · intro j hj
      exact boundsOK.2.2.1 (0 : Fin 3) (⟨j,hj⟩,()) trivial
    · exact boundsOK.2.2.1 (1 : Fin 3) () trivial
    · exact boundsOK.2.2.1 (2 : Fin 3) () trivial
  · intro vals initial binding
    let start := (job m (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) row).initial initial
    have px : ∀ j,j<N → SourceForwardGraph.row start stride j=Forward.value vals 0 j := by
      intro j hj
      simpa [SourceForwardGraph.row,start,job,Invocation.initial,bindRegisters,
        Forward.io,Forward.inputs,Forward.value,hj,mapValue] using
        binding (0 : Fin 3) (⟨j,hj⟩,()) trivial
    have pw : ∀ j,j<N → SourceForwardGraph.vector start "W" j=Forward.value vals 1 j := by
      intro j hj
      simpa [SourceForwardGraph.vector,start,job,Invocation.initial,bindRegisters,
        Forward.io,Forward.inputs,Forward.value,hj,mapValue] using
        binding (1 : Fin 3) (⟨j,hj⟩,()) trivial
    have pb : ∀ j,j<N → SourceForwardGraph.vector start "Bias" j=Forward.value vals 2 j := by
      intro j hj
      simpa [SourceForwardGraph.vector,start,job,Invocation.initial,bindRegisters,
        Forward.io,Forward.inputs,Forward.value,hj,mapValue] using
        binding (2 : Fin 3) (⟨j,hj⟩,()) trivial
    obtain ⟨t,ran,_,_,ys,mean,inverse,frame⟩ :=
      SourceForwardResult.executes m stride N B eps positive covers start
    obtain ⟨hme,hri,hys⟩ := graph_congr m N B eps _ _ _ _ _ _ px pw pb
    refine ⟨t,?_,?_,?_⟩
    · exact (runSchedule_singleton _ _).trans ran
    · intro o j _
      fin_cases o
      · change mapCell (observe "Y" (row*stride+j.1.val)) (t.mem "Y" (row*stride+j.1.val))=_
        have hj := ys j.1.val j.1.isLt
        change t.mem "Y" (row*stride+j.1.val)=_ at hj
        rw [hj,hys j.1.val j.1.isLt]
        rfl
      · change mapCell (observe "Mean" row) (t.mem "Mean" row)=_
        change t.mem "Mean" row=_ at mean
        rw [mean,hme]
        rfl
      · change mapCell (observe "Rstd" row) (t.mem "Rstd" row)=_
        change t.mem "Rstd" row=_ at inverse
        rw [inverse,hri]
        rfl
    · intro r off outside
      have out : SourceForwardResult.Outside stride N start r off := by
        refine ⟨?_,?_,?_⟩
        · exact outside.1 (1 : Fin 3) () trivial
        · exact outside.1 (2 : Fin 3) () trivial
        · by_cases hy : r≠"Y"
          · exact .inl hy
          · right
            intro j hj
            exact (outside.1 (0 : Fin 3) (⟨j,hj⟩,()) trivial).resolve_left hy
      exact frame r off out

#axiomsClean implements
end
end VeriTile.Bench.Repairs.LayerNormTriton.ForwardIO
