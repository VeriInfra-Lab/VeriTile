import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeTagsProgram
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FPReturned

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeTags
open VeriTile Triton FP.Structural
open FPStages (Stage)
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

private theorem fold_same {β : Type} (xs : List β) (p : β → Prop) [DecidablePred p]
    (d : TileDType) : xs.foldl (fun tag i => if p i then d else tag) d=d := by
  induction xs with
  | nil => rfl
  | cons i rest ih => simpa only [List.foldl_cons,ite_self] using ih

theorem output_preserves (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (c : OutputGrid.Shape) (scale : ℝ) (idx : GridIndex (OutputGrid.grid c))
    (s t : State α) (ran : (NativeLaunch.job M D (.output c scale) idx).run s=some t)
    (off : Nat) (before : cellDtype (s.mem "o" off)=.fp32) :
    cellDtype (t.mem "o" off)=.fp32 := by
  obtain ⟨final,exec,mem⟩ := NativeLaunch.program_run M D (.output c scale) idx s
  have same : t=final := Option.some.inj (ran.symm.trans exec)
  subst final
  rw [mem]
  change cellDtype ((FPOutput.written M D c scale (NativeLaunch.start (.output c scale) idx s)).mem "o" off)=.fp32
  rw [FPOutput.written,scatter_dtype]
  change (TileShape.allIndices [c.BT,c.BV]).foldl _ (cellDtype (s.mem "o" off))=.fp32
  rw [before]
  apply fold_same

theorem output_owner (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (c : OutputGrid.Shape) (scale : ℝ) (h : Fin c.heads) (row : Fin c.T) (v : Fin c.V)
    (s t : State α)
    (ran : (NativeLaunch.job M D (.output c scale) (OutputGrid.owner c h row v)).run s=some t) :
    cellDtype (t.mem "o" (h.val*(c.T*c.V)+row.val*c.V+v.val))=.fp32 := by
  let idx := OutputGrid.owner c h row v
  let initial := NativeLaunch.start (.output c scale) idx s
  let lane := OutputGrid.lane c row v
  have active (i : FPOutput.Index c) : FPOutput.active c initial i=
      VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputIO.active
        (OutputGrid.layout c) (OutputGrid.launchIndex c idx) i :=
    FPReturned.active_eq c (fun _ _=>(default : α)) idx i
  have address (i : FPOutput.Index c) : FPOutput.outputAddress c initial i=
      VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputIO.outputAddress
        (OutputGrid.layout c) (OutputGrid.launchIndex c idx) i :=
    FPReturned.address_eq c (fun _ _=>(default : α)) idx i
  obtain ⟨final,exec,mem⟩ := NativeLaunch.program_run M D (.output c scale) idx s
  have same : t=final := Option.some.inj (ran.symm.trans exec)
  subst final
  rw [mem]
  change cellDtype ((FPOutput.written M D c scale initial).mem "o" _)=.fp32
  have addr : FPOutput.outputAddress c initial lane=h.val*(c.T*c.V)+row.val*c.V+v.val :=
    (address lane).trans (OutputGrid.owner_address c h row v)
  rw [←addr,FPOutput.readback M D c scale initial lane (by
    rw [active]
    exact OutputGrid.owner_active c h row v) (by
      intro i j hi hj equal
      rw [active] at hi hj
      rw [address,address] at equal
      exact (OutputGrid.address_injective c idx idx i j hi hj equal).2)]
  rfl

private theorem output_jobs_preserve (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (c : OutputGrid.Shape) (scale : ℝ) (order : List (GridIndex (OutputGrid.grid c)))
    (s t : State α)
    (ran : runFillSchedule (order.map (fun idx=>.source (NativeLaunch.job M D (.output c scale) idx))) s=some t)
    (off : Nat) (before : cellDtype (s.mem "o" off)=.fp32) :
    cellDtype (t.mem "o" off)=.fp32 := by
  induction order generalizing s with
  | nil =>
    change some s=some t at ran
    rw [←Option.some.inj ran]
    exact before
  | cons idx rest ih =>
    change ((NativeLaunch.job M D (.output c scale) idx).run s).bind
      (runFillSchedule (rest.map (fun i=>.source (NativeLaunch.job M D (.output c scale) i))))=some t at ran
    cases first : (NativeLaunch.job M D (.output c scale) idx).run s with
    | none => simp [first] at ran
    | some next =>
      simp only [first,Option.bind_some] at ran
      exact ih next ran (output_preserves M D c scale idx s next first off before)

private theorem output_jobs_owner (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (c : OutputGrid.Shape) (scale : ℝ) (order : List (GridIndex (OutputGrid.grid c)))
    (h : Fin c.heads) (row : Fin c.T) (v : Fin c.V)
    (member : OutputGrid.owner c h row v∈order) (s t : State α)
    (ran : runFillSchedule (order.map (fun idx=>.source (NativeLaunch.job M D (.output c scale) idx))) s=some t) :
    cellDtype (t.mem "o" (h.val*(c.T*c.V)+row.val*c.V+v.val))=.fp32 := by
  induction order generalizing s with
  | nil => simp at member
  | cons idx rest ih =>
    change ((NativeLaunch.job M D (.output c scale) idx).run s).bind
      (runFillSchedule (rest.map (fun i=>.source (NativeLaunch.job M D (.output c scale) i))))=some t at ran
    cases first : (NativeLaunch.job M D (.output c scale) idx).run s with
    | none => simp [first] at ran
    | some next =>
      simp only [first,Option.bind_some] at ran
      rcases List.mem_cons.mp member with same|later
      · subst idx
        exact output_jobs_preserve M D c scale rest next t ran _
          (output_owner M D c scale h row v s next first)
      · exact ih later next ran

theorem consumer_dtype (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (c : Host.Dimensions) (scale : ℝ) (s t : State α)
    (ran : runFillSchedule (NativeLaunch.consumer M D c scale) s=some t)
    (off : Nat) (inside : off<Host.outputSize c) : cellDtype (t.mem "o" off)=.fp32 := by
  have positive : 0<Host.outputSize c := Nat.zero_lt_of_lt inside
  have tv : 0<c.T*c.V := by
    by_contra h
    have zero : c.T*c.V=0 := by omega
    simp [Host.outputSize,Nat.mul_assoc,zero] at positive
  have vp : 0<c.V := Nat.pos_of_mul_pos_left tv
  let h : Fin c.heads := ⟨off/(c.T*c.V),(Nat.div_lt_iff_lt_mul tv).mpr (by
    simpa only [Host.outputSize,Nat.mul_assoc] using inside)⟩
  let row : Fin c.T := ⟨(off%(c.T*c.V))/c.V,(Nat.div_lt_iff_lt_mul vp).mpr (Nat.mod_lt _ tv)⟩
  let v : Fin c.V := ⟨(off%(c.T*c.V))%c.V,Nat.mod_lt _ vp⟩
  have address : h.val*(c.T*c.V)+row.val*c.V+v.val=off := by
    dsimp only [h,row,v]
    rw [Nat.add_assoc,Nat.div_add_mod',Nat.div_add_mod']
  simp only [NativeLaunch.consumer,Nat.ne_of_gt positive,if_false] at ran
  have result := output_jobs_owner M D c.consumer scale (FPGrid.jobs (OutputGrid.grid c.consumer))
    h row v ((FPGrid.jobs_complete _).2 _) s t ran
  simpa only [Host.Dimensions.consumer,address] using result

#axiomsClean output_owner
#axiomsClean consumer_dtype
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeTags
