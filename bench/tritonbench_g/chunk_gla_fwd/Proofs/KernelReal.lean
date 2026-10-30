import bench.tritonbench_g.chunk_gla_fwd.Proofs.KernelTensor
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealWhole
import bench.tritonbench_g.chunk_gla_fwd.Proofs.OutputLogical

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.KernelReal
open VeriTile Triton KernelTensor
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open scoped BigOperators
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

/-- The mathematical input is arbitrary Real memory. Padding is supplied by
the selected zero-padding loads, not an assumption about input values. -/
def entry (m : RegionName → Nat → ℝ) : BlockState where
  mem := fun r i => .real (m r i)
  regs := fun _ _ _ => none
  pids := fun _ => 0

/-- Closed standalone consumer formula. A and history remain independent
caller inputs; no producer equality is required. -/
def consumerFormula (c : Host.Dimensions) (a : RealOutputIO.Inputs) (scale : ℝ)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) : ℝ :=
  (∑ k ∈ Finset.range c.K,
    a.query (Logical.featureAddress c.T c.K h.val t.val k) * scale *
      Real.exp (a.gate (Logical.featureAddress c.T c.K h.val t.val k)) *
      a.history (Geometry.historyAddress c.NT c.K c.V h.val (t.val/c.BT) k v.val)) +
  ∑ j ∈ Finset.range (t.val%c.BT+1),
    a.attention (h.val*c.T*c.BT+t.val*c.BT+j) *
      a.value (Geometry.outputAddress c.T c.V h.val (t.val/c.BT*c.BT+j) v.val)

theorem consumer_formula (c : Host.Dimensions) (a : RealOutputIO.Inputs) (scale : ℝ)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    OutputGrid.formula c.consumer a scale h t v=consumerFormula c a scale h t v := by
  have ht := t.isLt
  have hv := v.isLt
  have hmod := Nat.mod_lt t.val c.bt_pos
  have hbk : 0<c.outputBK := by
    apply lt_min (by decide)
    exact Host.nextPowerOfTwo_pos _ (lt_of_lt_of_le (by decide : 0<16) (Nat.le_max_left _ _))
  have hcausal (j : Nat) (hj : j ≤ t.val%c.BT) : t.val/c.BT*c.BT+j<c.T := by
    have hdiv := Nat.div_add_mod' t.val c.BT
    omega
  let historyTerm := fun k =>
    a.query (Logical.featureAddress c.T c.K h.val t.val k) * scale *
      Real.exp (a.gate (Logical.featureAddress c.T c.K h.val t.val k)) *
      a.history (Geometry.historyAddress c.NT c.K c.V h.val (t.val/c.BT) k v.val)
  let padded := fun k => if k<c.K then historyTerm k else 0
  have hsum : (∑ b : Fin (Geometry.ceilDiv c.K c.outputBK), ∑ e : Fin c.outputBK,
      padded (b.val*c.outputBK+e.val)) = ∑ k ∈ Finset.range c.K, historyTerm k := by
    calc
      _ = ∑ b ∈ Finset.range (Geometry.ceilDiv c.K c.outputBK), ∑ e : Fin c.outputBK,
          padded (b*c.outputBK+e.val) :=
        Fin.sum_univ_eq_sum_range (fun b => ∑ e : Fin c.outputBK, padded (b*c.outputBK+e.val)) _
      _ = ∑ k ∈ Finset.range c.K, padded k :=
        SumBlocks.truncated padded c.K c.outputBK hbk (by intro k hk; simp [padded,Nat.not_lt.mpr hk])
      _ = _ := by apply Finset.sum_congr rfl; intro k hk; simp [padded,Finset.mem_range.mp hk]
  unfold OutputGrid.formula RealOutputIO.formula
  simp only [OutputGrid.owner_launch,OutputGrid.lane,OutputGrid.layout,Host.Dimensions.consumer,
    Nat.div_add_mod',ht,hv,true_and,and_true,ite_true]
  unfold consumerFormula
  apply congrArg₂ (·+·)
  · calc
      _ = ∑ b : Fin (Geometry.ceilDiv c.K c.outputBK), ∑ e : Fin c.outputBK,
          padded (b.val*c.outputBK+e.val) := by
        apply Finset.sum_congr rfl
        intro b _
        apply Finset.sum_congr rfl
        intro e _
        by_cases hk : b.val*c.outputBK+e.val<c.K
        · simp only [hk,ite_true,padded,historyTerm,Logical.featureAddress,Host.Dimensions.NT]
          rw [Geometry.history_source_address]
        · simp [hk,padded]
      _ = _ := hsum
  · let f := fun j : Nat =>
      (if j ≤ t.val%c.BT then a.attention (h.val*c.T*c.BT+t.val*c.BT+j) else 0) *
        (if t.val/c.BT*c.BT+j<c.T then
          a.value (h.val*(c.T*c.V)+(t.val/c.BT*c.BT+j)*c.V+v.val) else 0)
    change (∑ j : Fin c.BT, f j.val) = _
    rw [Fin.sum_univ_eq_sum_range f c.BT]
    dsimp only [f]
    calc
      _ = ∑ j ∈ Finset.range (t.val%c.BT+1),
          (if j ≤ t.val%c.BT then a.attention (h.val*c.T*c.BT+t.val*c.BT+j) else 0) *
            (if t.val/c.BT*c.BT+j<c.T then
              a.value (h.val*(c.T*c.V)+(t.val/c.BT*c.BT+j)*c.V+v.val) else 0) := by
        symm
        apply Finset.sum_subset (Finset.range_mono (by omega))
        intro j _ hj
        simp only [Finset.mem_range,not_lt] at hj
        simp [show ¬j ≤ t.val%c.BT by omega]
      _ = _ := by
        apply Finset.sum_congr rfl
        intro j hj
        have hj' : j ≤ t.val%c.BT := by simpa [Nat.lt_succ_iff] using Finset.mem_range.mp hj
        simp only [hj',ite_true,hcausal j hj',Geometry.outputAddress,Nat.add_mul,Nat.mul_assoc]

def sourceExecution (mode : Admission.Mode) (c : Host.Dimensions) (scale : ℝ)
    (initial final : BlockState) : Prop :=
  match mode with
  | .producer => if c.K≤256 then
      Nonempty (Producer.DirectRun c.producer c.intraBK scale (RealWhole.attentionInit c initial) final)
    else Nonempty (Producer.LargeRun c.producer scale (Allocation.initialized c.producer initial) final)
  | .consumer => Nonempty (Kernel.GridLaunchedOrdinary (OutputGrid.kernel c.consumer scale)
      (OutputGrid.grid c.consumer) initial final)

def footprint (mode : Admission.Mode) (c : Host.Dimensions) : WriteFootprint :=
  match mode with
  | .producer => if c.K≤256 then Allocation.attention c.producer else
      fun cell => Allocation.attention c.producer cell ∨ Allocation.scratch (Producer.splitShape c.producer) cell
  | .consumer => OutputGrid.footprint c.consumer

def target (mode : Admission.Mode) (c : Host.Dimensions) (scale : ℝ) (initial : BlockState) : Index mode c → ℝ :=
  match mode with
  | .producer => fun i => Logical.attention c.T c.K c.BT (InterGrid.inputs initial) scale
      (head c i.1 i.2.1).val i.2.2.1.val i.2.2.2.val
  | .consumer => fun i => consumerFormula c (LocalReal.outputInputs OutputGrid.regions initial) scale
      (head c i.1 i.2.1) i.2.2.1 i.2.2.2

def Post (mode : Admission.Mode) (c : Host.Dimensions) (view : Output mode c device)
    (scale : ℝ) (initial final : BlockState) : Prop :=
  (∀ i,(realResult view final).get i=target mode c scale initial i) ∧
    BlockState.WriteWithin (footprint mode c) initial final

/-- Raw kernel output coordinates and frame, without any returned Tensor view. -/
def RawPost (mode : Admission.Mode) (c : Host.Dimensions)
    (scale : ℝ) (initial final : BlockState) : Prop :=
  (∀ i,final.readMem (region mode) (offset mode c i)=target mode c scale initial i) ∧
    BlockState.WriteWithin (footprint mode c) initial final

theorem raw_complete (mode : Admission.Mode) (c : Host.Dimensions)
    (scale : ℝ) (m : RegionName → Nat → ℝ) :
    ∃ final,sourceExecution mode c scale (entry m) final ∧ RawPost mode c scale (entry m) final := by
  cases mode with
  | producer =>
    have result := RealWhole.producer c scale (entry m) rfl
    by_cases small : c.K≤256
    · rw [if_pos small] at result
      obtain ⟨final,run,values,frame⟩ := result
      refine ⟨final,by simpa [sourceExecution,small] using run,?_,by simpa [footprint,small] using frame⟩
      intro i
      simp only [offset,region,target]
      exact values (head c i.1 i.2.1) i.2.2.1 i.2.2.2
    · rw [if_neg small] at result
      obtain ⟨final,run,values,frame⟩ := result
      refine ⟨final,by simpa [sourceExecution,small] using run,?_,by simpa [footprint,small] using frame⟩
      intro i
      simp only [offset,region,target]
      exact values (head c i.1 i.2.1) i.2.2.1 i.2.2.2
  | consumer =>
    obtain ⟨final,run,values,frame⟩ := RealWhole.consumer c scale (entry m) rfl
    refine ⟨final,run,?_,frame⟩
    intro i
    simp only [offset,region,target]
    have value := values (head c i.1 i.2.1) i.2.2.1 i.2.2.2
    rw [consumer_formula] at value
    simpa only [OutputGrid.returned,Host.Dimensions.consumer,Nat.mul_assoc] using value


theorem complete (mode : Admission.Mode) (c : Host.Dimensions) (view : Output mode c device)
    (scale : ℝ) (m : RegionName → Nat → ℝ) :
    ∃ final,sourceExecution mode c scale (entry m) final ∧ Post mode c view scale (entry m) final := by
  obtain ⟨final,ran,values,frame⟩ := raw_complete mode c scale m
  refine ⟨final,ran,?_,frame⟩
  simpa only [realResult,Tensor.get,coordinates] using values

#axiomsClean complete
#axiomsClean consumer_formula
#specNonCircular target avoiding [sourceExecution,FP.Structural.exec,exec,Producer.direct,Producer.large]
#specNonCircular Post avoiding [sourceExecution,FP.Structural.exec,exec,Producer.direct,Producer.large]
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.KernelReal
