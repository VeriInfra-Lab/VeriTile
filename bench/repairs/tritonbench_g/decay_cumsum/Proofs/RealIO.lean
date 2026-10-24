import bench.repairs.tritonbench_g.decay_cumsum.Proofs.RealAlloc

namespace VeriTile.Bench.Repairs.DecayCumsum.RealIO
open VeriTile.Triton Geometry Grid PackedStorage RealTiles
open VeriTile.Bench.TritonBenchG.DecayCumsum
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 5000000

def io (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : StreamGroupedEmitMasked3DKernelIO :=
  match kind with
  | .forward => {
      kernel := (forwardIO h p scale).kernel
      projection := (forwardIO h p scale).projection
      nIn := 1, nOut := 1, bufs := ["g","go"]
      inp := fun _ => "g", out := fun _ => "go"
      T := p.span, B := h.BK
      read := fun _ => (forwardIO h p scale).read1
      readMask := fun _ => (forwardIO h p scale).mask1
      write := fun _ => (forwardIO h p scale).write
      writeMask := fun _ => (forwardIO h p scale).writeMask }
  | .prepare => prepareIO h p scale
  | .backward => backwardIO h p scale

def values (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) :
    (Fin (io kind h p scale).nIn → Fin (io kind h p scale).T → Fin (io kind h p scale).B → ℝ) →
      Fin (io kind h p scale).nOut → Fin (io kind h p scale).T → Fin (io kind h p scale).B → ℝ :=
  match kind with
  | .forward => fun xs _ => ForwardCorrect.scan p.span h.BK (xs (0 : Fin 1))
  | .prepare => PrepareCorrect.values p.span h.BK scale
  | .backward => BackwardCorrect.values p.span h.BK

theorem source (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) :
    (io kind h p scale).kernel=(Kernels.source kind h p scale).eraseDType := by cases kind <;> rfl

theorem timeSize (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : (io kind h p scale).T=p.span := by cases kind <;> rfl

theorem laneSize (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : (io kind h p scale).B=h.BK := by cases kind <;> rfl

theorem buffers (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : (io kind h p scale).bufs=Roles.buffers kind := by cases kind <;> rfl

theorem input_member (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) (i : Fin (io kind h p scale).nIn) :
    (io kind h p scale).inp i ∈ Roles.buffers kind := by
  cases kind <;> fin_cases i <;> simp [io,prepareIO,backwardIO,PrepareCorrect.originalIO,BackwardCorrect.originalIO,Roles.buffers]

theorem output_member (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) (i : Fin (io kind h p scale).nOut) :
    (io kind h p scale).out i ∈ Roles.outputs kind := by
  cases kind <;> fin_cases i <;> simp [io,prepareIO,backwardIO,PrepareCorrect.originalIO,BackwardCorrect.originalIO,Roles.outputs]

theorem output_covers (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) (r : RegionName)
    (hr : r∈Roles.outputs kind) : ∃ i : Fin (io kind h p scale).nOut, (io kind h p scale).out i=r := by
  cases kind <;> simp only [Roles.outputs,List.mem_cons,List.not_mem_nil,or_false] at hr
  · subst r; exact ⟨⟨0,by simp [io,prepareIO,backwardIO,PrepareCorrect.originalIO,BackwardCorrect.originalIO]⟩,rfl⟩
  · rcases hr with rfl|rfl
    · exact ⟨⟨0,by simp [io,prepareIO,backwardIO,PrepareCorrect.originalIO,BackwardCorrect.originalIO]⟩,rfl⟩
    · exact ⟨⟨1,by simp [io,prepareIO,backwardIO,PrepareCorrect.originalIO,BackwardCorrect.originalIO]⟩,rfl⟩
  · rcases hr with rfl|rfl|rfl
    · exact ⟨⟨0,by simp [io,prepareIO,backwardIO,PrepareCorrect.originalIO,BackwardCorrect.originalIO]⟩,rfl⟩
    · exact ⟨⟨1,by simp [io,prepareIO,backwardIO,PrepareCorrect.originalIO,BackwardCorrect.originalIO]⟩,rfl⟩
    · exact ⟨⟨2,by simp [io,prepareIO,backwardIO,PrepareCorrect.originalIO,BackwardCorrect.originalIO]⟩,rfl⟩

theorem read_offset (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) (prog : Program h p)
    (i : Fin (io kind h p scale).nIn) (t : Fin (io kind h p scale).T) (j : Fin (io kind h p scale).B) :
    (io kind h p scale).read i prog.feature prog.chunk prog.row t j=relative h p prog t.val j.val := by cases kind <;> rfl

theorem write_offset (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) (prog : Program h p)
    (i : Fin (io kind h p scale).nOut) (t : Fin (io kind h p scale).T) (j : Fin (io kind h p scale).B) :
    (io kind h p scale).write i prog.feature prog.chunk prog.row t j=relative h p prog t.val j.val := by cases kind <;> rfl

theorem write_active (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) (prog : Program h p)
    (i : Fin (io kind h p scale).nOut) (t : Fin (io kind h p scale).T) (j : Fin (io kind h p scale).B) :
    (io kind h p scale).writeMask i prog.feature prog.chunk prog.row t j := by
  have bound := column_inside h p prog ⟨j.val,by simpa only [laneSize] using j.isLt⟩
  cases kind <;> exact bound

theorem implements (kind : Kind) (h : Call) (p : Phase) (hp : p∈phases h)
    (scale : ℝ) (positive : 0<h.DK) :
    StreamGroupedEmitMasked3DKernelIO.Implements (io kind h p scale) (fun _ _ _ => values kind h p scale) := by
  cases kind with
  | prepare => exact prepare_complete h p hp scale positive
  | backward => exact backward_complete h p hp scale positive
  | forward =>
    intro A hd hr hc p₀ p₁ p₂ xs u hp₀ hp₁ hp₂ hu hread hwrite hin
    obtain ⟨v,he,hv,hf⟩ := forward_complete h p scale positive A hd hr hc p₀ p₁ p₂ (xs (0 : Fin 1)) u
      hp₀ hp₁ hp₂ hu (hread (0 : Fin 1)) (hwrite (0 : Fin 1)) (hin (0 : Fin 1))
    refine ⟨v,he,?_,?_⟩
    · intro o t j mask
      exact hv t j mask
    · intro r o outside
      apply hf r o
      rcases outside with ne|out
      · exact Or.inl ne
      · exact Or.inr (out (0 : Fin 1))

#axiomsClean implements
end
end VeriTile.Bench.Repairs.DecayCumsum.RealIO
