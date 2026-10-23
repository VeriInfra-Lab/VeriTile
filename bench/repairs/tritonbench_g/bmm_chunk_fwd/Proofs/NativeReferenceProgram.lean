import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.NativeReference

namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeReference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open NativeGraph FPSuccess SourcePointers
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

def coordinatesFor (h : Host.Call) (c : Host.Config) (pids : Nat → Nat) : Coordinates :=
  ⟨pids 1,pids 2/h.shape.groups,pids 2-pids 2/h.shape.groups*h.shape.groups,
    pids 0/Host.cdiv h.shape.chunk c.N,pids 0%Host.cdiv h.shape.chunk c.N⟩

def activeBlock (h : Host.Call) (c : Host.Config) (d : Coordinates) : Bool :=
  !h.causal || decide (d.n*c.N<(d.m+1)*c.M)

/-- Closed per-tile numerical recurrence plus its masked destination scatter.
The old output is touched only by these exact source stores. -/
def evaluate (h : Host.Call) (c : Host.Config) (d : Coordinates)
    (memory : RegionName → Nat → Cell Expr) : RegionName → Nat → Cell Expr :=
  if activeBlock h c d then scatter h c d (capture h memory) memory else memory

theorem program_reference (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) (s : State Expr) :
    ∃ t,FP.Structural.exec (engine (backend rest c).algebra) (Kernels.source h c) s=some t ∧
      t.mem=evaluate h c (coordinatesFor h c s.pids) s.mem := by
  let d := coordinatesFor h c s.pids
  let x := capture h s.mem
  obtain ⟨entry,hpre,hbase,hmem,hactive,hcontrols⟩ := FPSuccess.prologue (backend rest c) h c s
  have controls : SourcePointers.Controls d entry := hcontrols
  have active_eq : activeBlock h c (coordinatesFor h c s.pids)=FPSuccess.active h c s := rfl
  cases act : FPSuccess.active h c s
  · refine ⟨entry,?_,?_⟩
    · change run _ (Kernels.source h c).surfaceBody s=some entry
      rw [source_split,run_append,hpre]
      simp only [Option.bind_some,run,step,hactive,act,Bool.false_eq_true,↓reduceIte]
      rfl
    · simpa only [evaluate,active_eq,act,Bool.false_eq_true,↓reduceIte] using hmem
  · obtain ⟨prepared,hsetup,hstate,hm0⟩ := SourcePointers.setup (backend rest c) h c d entry controls ha hb
    obtain ⟨prepared',hsetup',hzero⟩ := setup_zero rest h c entry hbase ha hb
    have same : prepared'=prepared := Option.some.inj (hsetup'.symm.trans hsetup)
    subst prepared'
    obtain ⟨terminal,hrange,hterminal,hm1,hacc⟩ := iterations_reference rest h c d
      0 (Host.cdiv h.shape.K c.K) (Nat.zero_le _) prepared hstate x
      (by rw [hm0,hmem]) hzero ha hb
    obtain ⟨final,hpost,hm2⟩ := epilogue_reference rest h c d terminal hterminal x
      (by rw [hm1,hm0,hmem]) hacc ha hb
    have hs : evalOp (engine (backend rest c).algebra) none (parts h c).start prepared=some (fun _ => (0 : Nat)) := by
      simp only [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize]; rfl
    have he : evalOp (engine (backend rest c).algebra) none (parts h c).stop prepared=some (fun _ => Host.cdiv h.shape.K c.K) := by
      simp only [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize]; rfl
    have hd : evalOp (engine (backend rest c).algebra) none (parts h c).stride prepared=some (fun _ => (1 : Nat)) := by
      simp only [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize]; rfl
    have hloop : step (engine (backend rest c).algebra)
        (.forRangeDyn "k" (parts h c).start (parts h c).stop (parts h c).stride (parts h c).body)
        prepared=some terminal := by
      simpa only [step,hs,he,hd,Option.bind_some] using hrange
    refine ⟨final,?_,?_⟩
    · change run _ (Kernels.source h c).surfaceBody s=some final
      rw [source_split,run_append,hpre]
      simp only [Option.bind_some,run,step,hactive,act,↓reduceIte]
      rw [branch_split,run_append,run_append,hsetup]
      simp only [Option.bind_some,run,hloop]
      simp [Option.bind,hpost]
    · rw [hm2,hm1,hm0,hmem]
      simp only [evaluate,active_eq,act,↓reduceIte]
      rfl

def grid (h : Host.Call) (c : Host.Config) (order : List (GridIndex (h.grid c)))
    (memory : RegionName → Nat → Cell Expr) :=
  order.foldl (fun m p => evaluate h c (coordinatesFor h c p.toPids) m) memory

theorem grid_reference (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32)
    (order : List (GridIndex (h.grid c))) (s : State Expr) :
    ∃ t,FPGrid.schedule (backend rest c) h c order s=some t ∧ t.mem=grid h c order s.mem := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons p ps ih =>
    obtain ⟨mid,hr,hm⟩ := program_reference rest h c ha hb (FPGrid.atIndex s p)
    obtain ⟨t,ht,hm'⟩ := ih mid
    refine ⟨t,by simp [FPGrid.schedule,hr,ht],?_⟩
    rw [hm',hm]
    rfl

#specNonCircular evaluate avoiding [Kernels.source,FP.Structural.exec,FP.Structural.run]
#specNonCircular grid avoiding [Kernels.source,FPGrid.schedule,FP.Structural.exec,FP.Structural.run]
#axiomsClean program_reference
#axiomsClean grid_reference
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeReference
