import bench.tritonbench_g.attention_llama.Proofs.RealFiniteContext

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealFiniteGeometry
open VeriTile Triton Source RealFiniteContext
open VeriTile.Bench.Repairs.AttentionLlama LoopInvariant HostGrid HostResult MaskCoverage Correctness
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
open Backing
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def program (B : Nat) (c : Config) (idx : GridIndex (RealIO.grid B c)) : Program B c.NH c.H :=
  ⟨idx ⟨0,by change 0<3; decide⟩,idx ⟨1,by change 1<3; decide⟩⟩

def gridIndex (B : Nat) (c : Config) (p : Program B c.NH c.H) : GridIndex (RealIO.grid B c) :=
  Fin.cases p.1 (Fin.cases p.2 (Fin.cases ⟨0,by change 0<1; decide⟩ (fun a=>Fin.elim0 a)))

@[simp] theorem program_gridIndex (B : Nat) (c : Config) (p : Program B c.NH c.H) :
    program B c (gridIndex B c p)=p := rfl

@[simp] theorem gridIndex_program (B : Nat) (c : Config) (idx : GridIndex (RealIO.grid B c)) :
    gridIndex B c (program B c idx)=idx := by
  funext a
  fin_cases a
  · rfl
  · rfl
  · apply Fin.ext
    have h:=(idx ⟨2,by change 2<3; decide⟩).isLt
    change (idx ⟨2,by change 2<3; decide⟩).val<1 at h
    change 0=(idx ⟨2,by change 2<3; decide⟩).val
    omega

theorem program_injective (B : Nat) (c : Config) : Function.Injective (program B c) := by
  intro i j h
  simpa only [gridIndex_program] using congrArg (gridIndex B c) h

theorem grid_pids (B : Nat) (c : Config) (idx : GridIndex (RealIO.grid B c)) :
    idx.toPids=fun ax=>if ax=0 then (program B c idx).1.val else
      if ax=1 then (program B c idx).2.val else 0 := by
  funext ax
  by_cases ha:ax<3
  · interval_cases ax
    · rfl
    · rfl
    · have hz:=Fin.eq_zero (show Fin 1 from idx ⟨2,by change 2<3; decide⟩)
      simpa only [GridIndex.toPids,RealIO.grid,Grid.rank,List.length_cons,List.length_nil,
        ↓reduceDIte,↓reduceIte] using congrArg Fin.val hz
  · simp only [GridIndex.toPids,RealIO.grid,Grid.rank,List.length_cons,List.length_nil,
      show ¬ax<0+1+1+1 from ha,↓reduceDIte,
      show ax≠0 by omega,show ax≠1 by omega,↓reduceIte]

/-- Old per-program contexts receive exactly the shared ordinary grid's
prepared predecessor, including its launch dimensions. -/
theorem context_grid (B : Nat) (c : Config) (scale : ℝ) (s : BlockState) (hD : 0<c.D)
    (idx : GridIndex (RealIO.grid B c)) :
    (contextFor (context c scale (s.withNumPids (RealIO.grid B c).toNumPids))
      (anchor hD (program B c idx))).initial=s.withGridIndex idx := by
  have h0:=congrArg (fun p=>p.1.val) (owner_anchor hD (program B c idx))
  have h1:=congrArg (fun p=>p.2.val) (owner_anchor hD (program B c idx))
  change (anchor hD (program B c idx)).query.val/64=(program B c idx).1.val at h0
  change (anchor hD (program B c idx)).batch.val*c.NH+
    (anchor hD (program B c idx)).head.val=(program B c idx).2.val at h1
  simp only [contextFor,context,BlockState.withNumPids,BlockState.withGridIndex,
    BlockState.withPids,grid_pids,h0,h1]

def writes (B : Nat) (c : Config) (p : Program B c.NH c.H) : WriteFootprint :=
  fun a=>∃ i : RealLayout.QueryIndex B c,owner (index i)=p ∧
    a.1=c.Out ∧ a.2=IOLayout.output c i

theorem writes_disjoint (B : Nat) (c : Config)
    (hinj : Function.Injective (IOLayout.output (B:=B) c))
    (p q : Program B c.NH c.H) (a : MemCellAddr)
    (hp : writes B c p a) (hq : writes B c q a) : p=q := by
  obtain ⟨i,hi,_,hio⟩:=hp
  obtain ⟨j,hj,_,hjo⟩:=hq
  have he:=hinj (hio.symm.trans hjo)
  subst j
  exact hi.symm.trans hj

theorem owner_writes (B : Nat) (c : Config) (i : RealLayout.QueryIndex B c) :
    writes B c (owner (index i)) (c.Out,IOLayout.output c i) := ⟨i,rfl,rfl,rfl⟩

theorem active_writes (B : Nat) (c : Config) (scale : ℝ) (s : BlockState) (hD : 0<c.D)
    (p : Program B c.NH c.H) (j : TileIndex [64,c.D])
    (active : alActive (contextFor (context c scale s) (anchor hD p)).initial c.H 64 j) :
    writes B c p (c.Out,alOutOffset (contextFor (context c scale s) (anchor hD p)).initial
      c.NH c.soz c.soh c.som c.son 64 j) := by
  have h0:=congrArg (fun q=>q.1.val) (owner_anchor hD p)
  change (anchor hD p).query.val/64=p.1.val at h0
  have hj : p.1.val*64+j.1.val<c.H := by
    change (anchor hD p).query.val/64*64+j.1.val<c.H at active
    rw [h0] at active
    exact active
  refine ⟨tile (laneIndex c hD p j hj),?_,rfl,?_⟩
  · rw [index_tile]
    exact owner_lane c hD p j hj
  · simpa only [config_context,tile,IOLayout.output,IOLayout.offset,Tensor.offset] using
      real_lane_address (context c scale s) c.Out c.soz c.soh c.som c.son c.fp8 hD p j hj

def outputLayout (B : Nat) (c : Config) (scale : ℝ) (s : BlockState)
    (hinj : Function.Injective (IOLayout.output (B:=B) c)) : OutputLayout B (context c scale s) where
  base:=0
  soz:=c.soz
  soh:=c.soh
  som:=c.som
  son:=c.son
  injective:=by
    intro i j he
    have eq:=hinj (show IOLayout.output c (tile i)=IOLayout.output c (tile j) from by
      simpa only [outputAddress_eq,Nat.zero_add,IOLayout.output,IOLayout.offset,tile,context] using he)
    simpa only [index_tile] using congrArg index eq

end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealFiniteGeometry
