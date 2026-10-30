import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.FPBody

namespace VeriTile.Bench.Repairs.BmmChunkFwd.FPSuccess
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem iterations {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (cur stop : Nat) (s : State α) (ready : Ready h c s) :
    ∃ t, range (engine o.algebra) "k" cur stop 1 (parts h c).body s=some t ∧
      Ready h c t ∧ t.mem=s.mem ∧ Fixed s t := by
  by_cases less : cur<stop
  · obtain ⟨mid,hm,hr,hmem,hfixed⟩ := body o h c s ready cur
    obtain ⟨t,ht,htr,htm,htfixed⟩ := iterations o h c (cur+1) stop mid hr
    refine ⟨t,?_,htr,htm.trans hmem,hfixed.trans htfixed⟩
    rw [range,if_neg (by decide),if_pos less,hm]
    exact ht
  · exact ⟨s,by rw [range,if_neg (by decide),if_neg less],ready,rfl,Fixed.refl s⟩
termination_by stop-cur

theorem loop_success {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (s : State α) (ready : Ready h c s) :
    ∃ t, step (engine o.algebra)
      (.forRangeDyn "k" (parts h c).start (parts h c).stop (parts h c).stride (parts h c).body) s=some t ∧
      Ready h c t ∧ t.mem=s.mem ∧ Fixed s t := by
  obtain ⟨t,ht,hr,hm,hfixed⟩ := iterations o h c 0 (Host.cdiv h.shape.K c.K) s ready
  refine ⟨t,?_,hr,hm,hfixed⟩
  have hs : FP.Structural.evalOp (engine o.algebra) none (parts h c).start s=some (fun _ => (0 : Nat)) := by
    cases ha : h.aStorage <;> cases hb : h.bStorage <;>
      simp only [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize] <;> rfl
  have he : FP.Structural.evalOp (engine o.algebra) none (parts h c).stop s=some (fun _ => Host.cdiv h.shape.K c.K) := by
    cases ha : h.aStorage <;> cases hb : h.bStorage <;>
      simp only [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize] <;> rfl
  have hd : FP.Structural.evalOp (engine o.algebra) none (parts h c).stride s=some (fun _ => (1 : Nat)) := by
    cases ha : h.aStorage <;> cases hb : h.bStorage <;>
      simp only [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize] <;> rfl
  simpa only [step,hs,he,hd,Option.bind_some] using ht

def row {α : Type} (h : Host.Call) (c : Host.Config) (s : State α) (i : TileIndex [c.M,c.N]) :=
  s.pids 0/Host.cdiv h.shape.chunk c.N*c.M+i.1.val

def column {α : Type} (h : Host.Call) (c : Host.Config) (s : State α) (i : TileIndex [c.M,c.N]) :=
  s.pids 0%Host.cdiv h.shape.chunk c.N*c.N+i.2.1.val

def offset {α : Type} (h : Host.Call) (c : Host.Config) (s : State α) (i : TileIndex [c.M,c.N]) :=
  (s.pids 1*((h.shape.chunks*h.shape.groups)*h.shape.chunk*h.shape.chunk)+
    s.pids 2/h.shape.groups*(h.shape.groups*h.shape.chunk*h.shape.chunk)+
    (s.pids 2-s.pids 2/h.shape.groups*h.shape.groups)*
      (if h.shape.grouped then h.shape.chunk*h.shape.chunk else 0))+
    (h.shape.chunk*row h c s i+column h c s i*1)

def laneMask {α : Type} (h : Host.Call) (c : Host.Config) (s : State α) (i : TileIndex [c.M,c.N]) : Bool :=
  decide (row h c s i<h.shape.chunk) && decide (column h c s i<h.shape.chunk)

def Writes {α : Type} (h : Host.Call) (c : Host.Config) (s : State α) (rg : RegionName) (off : Nat) : Prop :=
  ∃ i, laneMask h c s i=Bool.true ∧ rg=h.Out ∧ off=offset h c s i

def ProgramWrites {α : Type} (h : Host.Call) (c : Host.Config) (s : State α) (rg : RegionName) (off : Nat) : Prop :=
  active h c s=Bool.true ∧ Writes h c s rg off

def RegionFrame {α : Type} (out : RegionName) (s t : State α) : Prop :=
  ∀ rg off, rg≠out → t.mem rg off=s.mem rg off

end
end VeriTile.Bench.Repairs.BmmChunkFwd.FPSuccess

#guard_msgs (drop info) in
#auditModuleAxioms
