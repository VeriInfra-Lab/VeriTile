import bench.tritonbench_g.triton_attention.Proofs.RealForwardGrid
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def SavedAt (c : Shape) (scale : ℝ) (μ : RegionName → Nat → ℝ) (i : Coord c) (mem : Grid.Memory R) : Prop :=
  ∃ m,mem "M" (rowOffset c i.1 i.2.1)=.mk .fp32 (some m) ∧
    mem "L" (rowOffset c i.1 i.2.1)=.mk .fp32 (some
      (Mathematical.mass i.2.1.val c.rows
        (Mathematical.score c.dim scale (tensorAt c μ "Q" i.1.val) (tensorAt c μ "K" i.1.val) i.2.1.val) m))

theorem forwardStep_saved (c : Shape) (scale : ℝ) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) (h : ForwardInputs mem μ) (i : Coord c) :
    SavedAt c scale μ i (Grid.forwardStep model c scale (Grid.forwardProgram c i) mem) := by
  let pid:=Grid.forwardProgram c i
  let s:=Grid.entry (Grid.forwardGrid c) pid mem
  let row : FRow (forwardParameters c scale):=((Grid.forwardLane c scale i).1,())
  have hp:=Grid.forwardProgram_legal c i
  have h0 : s.pids 0<c.blocks := hp.1
  have h1 : s.pids 1<c.groups := hp.2
  obtain ⟨final,he,hm,_,hl,hmax,_⟩:=forwardCorrectness algebra model (forwardParameters c scale) s
    (by change 0<128;decide) (by dsimp [forwardStop,forwardParameters];exact dvd_mul_left 128 _)
    (by simp [forwardParameters]) (by simp [forwardParameters]) (by simp [forwardParameters]) (forward_offset_injective _ _)
  obtain ⟨m,hM,hL⟩:=forward_saved c scale s μ (inputs_forwardReads c scale pid mem μ h) h0 h1 row
  refine ⟨m,?_,?_⟩
  · have ans:=(hmax row).trans hM
    rw [hm] at ans
    change Grid.forwardStep model c scale (Grid.forwardProgram c i) mem "M"
      (forwardRowOffset (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) (Grid.forwardProgram c i) mem)
        ((Grid.forwardLane c scale i).1,()))=_ at ans
    rw [Grid.forward_target_row] at ans
    exact ans
  · have ans:=(hl row).trans hL
    rw [hm] at ans
    change Grid.forwardStep model c scale (Grid.forwardProgram c i) mem "L"
      (forwardRowOffset (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) (Grid.forwardProgram c i) mem)
        ((Grid.forwardLane c scale i).1,()))=_ at ans
    rw [Grid.forward_target_row] at ans
    have hscore : logicalScore (forwardParameters c scale) s μ row.1=
        Mathematical.score c.dim scale (tensorAt c μ "Q" i.1.val) (tensorAt c μ "K" i.1.val) i.2.1.val := by
      funext j
      change Mathematical.score c.dim scale
        (qAt (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid mem) μ)
        (kAt (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid mem) μ)
        (pid.1*128+(Grid.forwardLane c scale i).1.val) j=_
      rw [qAt_entry,kAt_entry]
      simp [pid,Grid.forwardProgram,Grid.forwardLane,Nat.div_add_mod']
    rw [hscore] at ans
    simpa [row,tensorAt,logicalScore,qAt,kAt,forwardQueryRow,forwardHeadRow,forwardParameters,s,pid,
      Grid.entry,Grid.ids,Grid.forwardProgram,Grid.forwardLane,c.dim_pos,Nat.div_add_mod'] using ans

theorem fold_achieves {α : Type} (step : Grid.Program → Grid.Memory α → Grid.Memory α)
    (inv post : Grid.Memory α → Prop) (work : List Grid.Program) (mem : Grid.Memory α)
    (target : Grid.Program) (ht : target∈work) (hm : inv mem)
    (hs : ∀ pid∈work,∀ m,inv m → inv (step pid m))
    (hw : ∀ m,inv m → post (step target m))
    (hf : ∀ pid∈work,pid≠target → ∀ m,post m → post (step pid m)) :
    post (work.foldl (fun m pid=>step pid m) mem) := by
  induction work generalizing mem with
  | nil=>simp at ht
  | cons p ps ih=>
    by_cases hh : target∈ps
    · exact ih (step p mem) hh (hs p (by simp) mem hm)
        (by intro pid hp m h;exact hs pid (by simp [hp]) m h)
        (by intro pid hp hn m h;exact hf pid (by simp [hp]) hn m h)
    · have he : p=target := by simpa [hh,eq_comm] using ht
      rw [List.foldl_cons,he]
      exact Grid.fold_invariant step post ps (step target mem) (hw mem hm)
        (by intro pid hp m h;exact hf pid (by simp [hp]) (by intro he;subst pid;exact hh hp) m h)

theorem forward_grid_saved (c : Shape) (scale : ℝ) (work : List Grid.Program)
    (schedule : Grid.FullSchedule (Grid.forwardGrid c) work) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) (h : ForwardInputs mem μ) (i : Coord c) :
    SavedAt c scale μ i (work.foldl (fun m pid=>Grid.forwardStep model c scale pid m) mem) := by
  apply fold_achieves (Grid.forwardStep model c scale) (fun m=>ForwardInputs m μ)
    (SavedAt c scale μ i) work mem (Grid.forwardProgram c i)
    (Grid.FullSchedule.covers (Grid.forwardGrid c) work schedule _ (Grid.forwardProgram_legal c i)) h
  · intro pid hp m hm
    exact forwardStep_inputs c scale pid m μ hm (Grid.FullSchedule.legal (Grid.forwardGrid c) work schedule pid hp)
  · intro m hm
    exact forwardStep_saved c scale m μ hm i
  · intro pid hp hn m hm
    obtain ⟨mx,hM,hL⟩:=hm
    refine ⟨mx,?_,?_⟩
    · rw [Grid.forwardStep_otherRow model c scale pid m (Grid.FullSchedule.legal (Grid.forwardGrid c) work schedule pid hp) i hn "M" (Or.inr rfl)]
      exact hM
    · rw [Grid.forwardStep_otherRow model c scale pid m (Grid.FullSchedule.legal (Grid.forwardGrid c) work schedule pid hp) i hn "L" (Or.inl rfl)]
      exact hL

#axiomsClean forward_grid_saved
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
