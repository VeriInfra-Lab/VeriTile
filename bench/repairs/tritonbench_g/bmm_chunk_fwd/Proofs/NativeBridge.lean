import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.NativeControl

namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeControl
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section

theorem control_values (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (metadata : Metadata h c) (p : Geometry.PID h c) :
    (b h c p).toNat=Geometry.p1 p ∧
    (ch h c p).toNat=Geometry.p2 p/h.shape.groups ∧
    (head h c p).toNat=Geometry.p2 p-Geometry.p2 p/h.shape.groups*h.shape.groups ∧
    (m h c p).toNat=Geometry.p0 p/Host.cdiv h.shape.chunk c.N ∧
    (n h c p).toNat=Geometry.p0 p%Host.cdiv h.shape.chunk c.N ∧
    (ceil h.shape.K c.K).toNat=Host.cdiv h.shape.K c.K := by
  obtain ⟨hb,hch,hh,hm,hn,hk⟩ := controls h c hc metadata p
  rw [hb,hch,hh,hm,hn,hk]
  have hp0 : Geometry.p0 p<9223372036854775808 := by
    have ht := (p ⟨0,by change 0<3; decide⟩).isLt
    change Geometry.p0 p<Host.cdiv h.shape.chunk c.M*Host.cdiv h.shape.chunk c.N at ht
    have := metadata.gridX; omega
  have hp1 : Geometry.p1 p<9223372036854775808 := by
    have ht := (p ⟨1,by change 1<3; decide⟩).isLt
    change Geometry.p1 p<h.shape.batch at ht
    have := metadata.gridY; omega
  have hp2 : Geometry.p2 p<9223372036854775808 := by
    have := Geometry.p2_lt p; have := metadata.gridZ; omega
  have hK := (Finite.dimensions_bounded h.shape metadata.sizes).2.2
  have hck := (Finite.configuration_small c hc).2.2
  have hstop : Host.cdiv h.shape.K c.K<9223372036854775808 := by
    have := Nat.div_le_self (h.shape.K+c.K-1) c.K
    unfold Host.cdiv; omega
  exact ⟨nat64 _ hp1,nat64 _ (lt_of_le_of_lt (Nat.div_le_self _ _) hp2),
    nat64 _ (lt_of_le_of_lt (Nat.sub_le _ _) hp2),
    nat64 _ (lt_of_le_of_lt (Nat.div_le_self _ _) hp0),
    nat64 _ (lt_of_le_of_lt (Nat.mod_le _ _) hp0),nat64 _ hstop⟩

def active (h : Host.Call) (c : Host.Config) (p : Geometry.PID h c) : Bool :=
  !h.causal || (n h c p*BitVec.ofNat 64 c.N).slt ((m h c p+1)*BitVec.ofNat 64 c.M)

def Controls {α : Type} (h : Host.Call) (c : Host.Config) (p : Geometry.PID h c) (t : State α) : Prop :=
  t.regs .nat [] "pid_b"=some (fun _ => (b h c p).toNat) ∧
  t.regs .nat [] "pid_c"=some (fun _ => (ch h c p).toNat) ∧
  t.regs .nat [] "pid_h"=some (fun _ => (head h c p).toNat) ∧
  t.regs .nat [] "pid_m"=some (fun _ => (m h c p).toNat) ∧
  t.regs .nat [] "pid_n"=some (fun _ => (n h c p).toNat)

/-- The actual source prologue executes with exactly the native signed-cast,
signed quotient/remainder and ceil controls. Its evaluated causal guard agrees
with the actual signed-i64 native comparison. -/
theorem source_prologue {α : Type} [Inhabited α] (o : FPSuccess.Ops α)
    (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (metadata : Metadata h c) (p : Geometry.PID h c) (s : State α) :
    ∃ t, run (engine o.algebra) (FPSuccess.branch h c).before (FPGrid.atIndex s p)=some t ∧
      Controls h c p t ∧
      evalOp (engine o.algebra) none (FPSuccess.branch h c).condition t=some (fun _ => active h c p) := by
  obtain ⟨t,hrun,_,_,hactive,hcontrols⟩ := FPSuccess.prologue o h c (FPGrid.atIndex s p)
  obtain ⟨hb,hch,hh,hm,hn,hk⟩ := control_values h c hc metadata p
  obtain ⟨_,_,_,hm',hn',_⟩ := controls h c hc metadata p
  refine ⟨t,hrun,?_,?_⟩
  · simpa only [Controls,hb,hch,hh,hm,hn,FPSuccess.Controls,
      FPGrid.atIndex,Geometry.p0,Geometry.p1,Geometry.p2,GridIndex.toPids,
      Host.Call.grid,Grid.rank,List.length_cons,List.length_nil,↓reduceDIte] using hcontrols
  · have hcausal := (Finite.source_indices h c hc metadata.sizes).causal p
    rw [active,hm',hn',hcausal]
    simpa only [FPSuccess.active,FPGrid.atIndex,Geometry.p0,GridIndex.toPids,
      Host.Call.grid,Grid.rank,List.length_cons,List.length_nil,↓reduceDIte] using hactive

/-- The actual source's K-loop stop evaluation equals native ceildiv, rather
than merely placing an unrelated integer bound beside a floating execution. -/
theorem source_loop_stop {α : Type} [Inhabited α] (o : FPSuccess.Ops α)
    (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (metadata : Metadata h c) (p : Geometry.PID h c) (s : State α) :
    evalOp (engine o.algebra) none (FPSuccess.parts h c).stop s=
      some (fun _ => (ceil h.shape.K c.K).toNat) := by
  rw [(control_values h c hc metadata p).2.2.2.2.2]
  cases ha : h.aStorage <;> cases hb : h.bStorage <;>
    simp only [FPSuccess.parts,FPSuccess.branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize] <;> rfl

#axiomsClean source_prologue
#axiomsClean source_loop_stop
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeControl

#guard_msgs (drop info) in
#auditModuleAxioms
