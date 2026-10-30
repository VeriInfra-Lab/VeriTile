import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Target
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.AddressBounds

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.RealFormulaBinding
open VeriTile Triton Whole AddressBounds
noncomputable section
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false

def ReadAgreement (c : Config) (μ ν : RegionName → Nat → ℝ) : Prop :=
  ∀ r,r∈(["q","k","v","h","g","do","dh"] : List RegionName) →
    ∀ i,i<extent c r → μ r i=ν r i

theorem logical_inputs (c : Config) (order : List (CTA c)) (s : BlockState)
    (vals : IO.Inputs ℝ c) (hin : (IO.realIO c order).InputBinding s vals) :
    ReadAgreement c s.readMem (Target.inputs c vals) := by
  have read (i : Fin 7) (n : Nat) (hn : n<IO.inputSize c i) :
      s.readMem (IO.inputRegion i) n=vals i ⟨n,hn⟩ := hin i ⟨n,hn⟩ trivial
  intro r hr n hn
  simp only [List.mem_cons,List.mem_nil_iff,or_false] at hr
  rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp only [Target.inputs,ite_true,ite_false,reduceCtorEq]
  · simpa [Target.get,IO.inputSize,IO.inputRegion,hn] using read 0 n hn
  · simpa [Target.get,IO.inputSize,IO.inputRegion,hn] using read 1 n hn
  · simpa [Target.get,IO.inputSize,IO.inputRegion,hn] using read 2 n hn
  · simpa [Target.get,IO.inputSize,IO.inputRegion,hn] using read 3 n hn
  · simpa [Target.get,IO.inputSize,IO.inputRegion,hn] using read 4 n hn
  · simpa [Target.get,IO.inputSize,IO.inputRegion,hn] using read 5 n hn
  · simpa [Target.get,IO.inputSize,IO.inputRegion,hn] using read 6 n hn

theorem formula (c : Config) (μ ν : RegionName → Nat → ℝ) (reads : ReadAgreement c μ ν)
    (a : CTA c) (r e : Nat) :
    Spec.dq c (Spec.views c μ) a.head.val a.time.val r e=
      Spec.dq c (Spec.views c ν) a.head.val a.time.val r e ∧
    Spec.dk c (Spec.views c μ) a.head.val a.time.val r e=
      Spec.dk c (Spec.views c ν) a.head.val a.time.val r e ∧
    Spec.dg c (Spec.views c μ) a.key.val a.head.val a.time.val r=
      Spec.dg c (Spec.views c ν) a.key.val a.head.val a.time.val r := by
  have cap : Capacity c (extent c):=fun _=>le_rfl
  have qk (which : RegionName) (hw : which="q" ∨ which="k") (row key : Nat) :
      Spec.qk c (fun h t k=>μ which ((h*c.T+t)*c.K+k)) a.head.val a.time.val row key=
        Spec.qk c (fun h t k=>ν which ((h*c.T+t)*c.K+k)) a.head.val a.time.val row key := by
    simp only [Spec.qk]
    split_ifs with active
    · apply reads which (by rcases hw with rfl|rfl <;> simp)
      simpa [Nat.add_mul,Nat.mul_assoc] using
        query_bound c a (extent c) cap which (by tauto) _ _ active.1 active.2
    · rfl
  have vd (which : RegionName) (hw : which="v" ∨ which="do") (row key : Nat) :
      Spec.vd c (fun h t k=>μ which ((h*c.T+t)*c.V+k)) a.head.val a.time.val row key=
        Spec.vd c (fun h t k=>ν which ((h*c.T+t)*c.V+k)) a.head.val a.time.val row key := by
    simp only [Spec.vd]
    split_ifs with active
    · apply reads which (by rcases hw with rfl|rfl <;> simp)
      simpa [Nat.add_mul,Nat.mul_assoc] using
        value_bound c a (extent c) cap which hw _ _ active.1 active.2
    · rfl
  have hist (which : RegionName) (hw : which="h" ∨ which="dh") (key j : Nat) :
      Spec.hist c (fun h t k v=>μ which (((h*c.NT+t)*c.K+k)*c.V+v)) a.head.val a.time.val key j=
        Spec.hist c (fun h t k v=>ν which (((h*c.NT+t)*c.K+k)*c.V+v)) a.head.val a.time.val key j := by
    simp only [Spec.hist]
    split_ifs with active
    · apply reads which (by rcases hw with rfl|rfl <;> simp)
      simpa [Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using
        history_bound c a (extent c) cap which hw _ _ active.1 active.2
    · rfl
  have gate (row : Nat) : Spec.gate c (Spec.views c μ) a.head.val a.time.val row=
      Spec.gate c (Spec.views c ν) a.head.val a.time.val row := by
    simp only [Spec.gate,Spec.views]
    split_ifs with active
    · exact reads "g" (by simp) _ (gate_bound c a (extent c) cap _ active)
    · rfl
  have last : Spec.lastGate c (Spec.views c μ) a.head.val a.time.val=
      Spec.lastGate c (Spec.views c ν) a.head.val a.time.val := by
    apply reads "g" (by simp)
    apply gate_bound c a (extent c) cap
    have := T_pos c a
    omega
  have hq:=qk "q" (Or.inl rfl)
  have hk:=qk "k" (Or.inr rfl)
  have hv:=vd "v" (Or.inl rfl)
  have hd:=vd "do" (Or.inr rfl)
  have hh:=hist "h" (Or.inl rfl)
  have hf:=hist "dh" (Or.inr rfl)
  have bq (row key : Nat) : Spec.baseQ c (Spec.views c μ) a.head.val a.time.val row key=
      Spec.baseQ c (Spec.views c ν) a.head.val a.time.val row key := by
    simp only [Spec.baseQ]
    rw [gate]
    simp only [Spec.views,hv,hd,hh,hf]
  have bk (row key : Nat) : Spec.baseK c (Spec.views c μ) a.head.val a.time.val row key=
      Spec.baseK c (Spec.views c ν) a.head.val a.time.val row key := by
    simp only [Spec.baseK]
    rw [gate,last]
    simp only [Spec.views,hv,hd,hh,hf]
  have score (row key : Nat) : Spec.score c (Spec.views c μ) a.head.val a.time.val row key=
      Spec.score c (Spec.views c ν) a.head.val a.time.val row key := by
    simp only [Spec.score]
    simp_rw [gate]
    simp only [Spec.views,hv,hd]
  have dq (row key : Nat) : Spec.dq c (Spec.views c μ) a.head.val a.time.val row key=
      Spec.dq c (Spec.views c ν) a.head.val a.time.val row key := by
    simp only [Spec.dq]
    rw [bq]
    simp_rw [score]
    simp only [Spec.views,hk]
  have dk (row key : Nat) : Spec.dk c (Spec.views c μ) a.head.val a.time.val row key=
      Spec.dk c (Spec.views c ν) a.head.val a.time.val row key := by
    simp only [Spec.dk]
    rw [bk]
    simp_rw [score]
    simp only [Spec.views,hq]
  refine ⟨dq r e,dk r e,?_⟩
  simp only [Spec.dg]
  simp_rw [dq,dk]
  simp only [Spec.correction]
  rw [last]
  simp_rw [bk]
  simp only [Spec.views,hq,hk,hh,hf]

end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.RealFormulaBinding
