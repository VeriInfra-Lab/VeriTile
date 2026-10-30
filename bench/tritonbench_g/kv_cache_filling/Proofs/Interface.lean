import bench.tritonbench_g.kv_cache_filling.Proofs.RowCongruence
import bench.tritonbench_g.kv_cache_filling.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.KvCacheFilling.Interface
open VeriTile.Triton FP.Structural QuantProfile CopyExecution QuantExecution
noncomputable section
set_option maxHeartbeats 3000000

@[simp] theorem initial_pid {α : Type} (a : Ops α) (r : Request) (v : Bool) (pid : Nat)
    (s : State α) : ((invocation a r v pid).initial s).pids 0 = pid := rfl

theorem input_source {α : Type} [Inhabited α] (a : Ops α) (r : Request) (j : Job)
    (v : Bool) (pid : Nat) (vals : Inputs α r) (dom : inputDomain r j vals) (s : State α)
    (hin : (programIO a r j v pid).InputBinding vals s) :
    (((invocation a r v pid).initial s).mem "src" pid).read .int = (sourceRow r j : Int) := by
  have h := hin ⟨0,by simp [programIO,memory,inputs]⟩ PUnit.unit True.intro
  exact h.trans dom.1

theorem input_destination {α : Type} [Inhabited α] (a : Ops α) (r : Request) (j : Job)
    (v : Bool) (pid : Nat) (vals : Inputs α r) (dom : inputDomain r j vals) (s : State α)
    (hin : (programIO a r j v pid).InputBinding vals s) :
    (((invocation a r v pid).initial s).mem "dst" pid).read .int = (destinationRow r j : Int) := by
  have h := hin ⟨1,by simp [programIO,memory,inputs]⟩ PUnit.unit True.intro
  exact h.trans dom.2

theorem input_key {α : Type} [Inhabited α] (a : Ops α) (r : Request) (j : Job)
    (v : Bool) (pid : Nat) (vals : Inputs α r) (s : State α)
    (hin : (programIO a r j v pid).InputBinding vals s) (i : Nat) (hi : i < r.dk) :
    inputRow (Shared.parameters r) ((invocation a r v pid).initial s)
      (sourceRow r j) Bool.true i = keyRow r vals i := by
  have h := hin ⟨2,by simp [programIO,memory,inputs]⟩ (⟨i,hi⟩,PUnit.unit) True.intro
  simpa [keyRow,hi,inputRow,width,input,Shared.parameters,programIO,memory,inputAddress,inputs,
    mapValue] using h

theorem input_value {α : Type} [Inhabited α] (a : Ops α) (r : Request) (j : Job)
    (v : Bool) (pid : Nat) (vals : Inputs α r) (s : State α)
    (hin : (programIO a r j v pid).InputBinding vals s) (i : Nat) (hi : i < r.dv) :
    inputRow (Shared.parameters r) ((invocation a r v pid).initial s)
      (sourceRow r j) Bool.false i = valueRow r vals i := by
  have h := hin ⟨3,by simp [programIO,memory,inputs]⟩ (⟨i,hi⟩,PUnit.unit) True.intro
  simpa [valueRow,hi,inputRow,width,input,Shared.parameters,programIO,memory,inputAddress,inputs,
    mapValue] using h

theorem source_safe {α : Type} [Inhabited α] (a : Ops α) (r : Request) (d : Domain r)
    (j : Job) (v : Bool) (pid : Nat) (vals : Inputs α r) (dom : inputDomain r j vals)
    (bounds : RegionName → Nat) (hb : (programIO a r j v pid).InBounds bounds vals)
    (s : State α) (hin : (programIO a r j v pid).InputBinding vals s) :
    (invocation a r v pid).Safe bounds s := by
  let t := (invocation a r v pid).initial s
  have hs : (t.mem "src" (t.pids 0)).read .int = (sourceRow r j : Int) :=
    input_source a r j v pid vals dom s hin
  have hd : (t.mem "dst" (t.pids 0)).read .int = (destinationRow r j : Int) :=
    input_destination a r j v pid vals dom s hin
  have bs : t.pids 0 < bounds "src" := hb.2.1 ⟨0,by simp [programIO,memory,inputs]⟩ PUnit.unit True.intro
  have bd : t.pids 0 < bounds "dst" := hb.2.1 ⟨1,by simp [programIO,memory,inputs]⟩ PUnit.unit True.intro
  have bk : ∀ i, i < r.dk → sourceRow r j*r.dk+i < bounds "key" := by
    intro i hi
    exact hb.2.1 ⟨2,by simp [programIO,memory,inputs]⟩ (⟨i,hi⟩,PUnit.unit) True.intro
  have bv : ∀ i, i < r.dv → sourceRow r j*r.dv+i < bounds "value" := by
    intro i hi
    exact hb.2.1 ⟨3,by simp [programIO,memory,inputs]⟩ (⟨i,hi⟩,PUnit.unit) True.intro
  have bkc : ∀ i, i < r.ck → destinationRow r j*r.ck+i < bounds "keyCache" := by
    intro i hi
    exact hb.2.2.1 ⟨0,by simp [programIO,memory,outputs]⟩ (⟨i,hi⟩,PUnit.unit) True.intro
  have bvc : ∀ i, i < r.cv → destinationRow r j*r.cv+i < bounds "valueCache" := by
    intro i hi
    exact hb.2.2.1 ⟨1,by simp [programIO,memory,outputs]⟩ (⟨i,hi⟩,PUnit.unit) True.intro
  have bkm : r.policy ≠ .copy → ∀ i, i < 2 → destinationRow r j*2+i < bounds "keyMeta" := by
    intro hp i hi
    exact hb.2.2.1 ⟨2,by simp [programIO,memory,outputs]⟩ (⟨i,hi⟩,PUnit.unit) hp
  have bvm : r.policy ≠ .copy → ∀ i, i < 2 → destinationRow r j*2+i < bounds "valueMeta" := by
    intro hp i hi
    exact hb.2.2.1 ⟨3,by simp [programIO,memory,outputs]⟩ (⟨i,hi⟩,PUnit.unit) hp
  have pos := SharedExecution.positive r d
  have widths := packed_widths_positive r d
  change FP.Structural.TraceSafe (algebra a) bounds (Shared.kernel v r) t
  cases hp : r.policy with
  | copy =>
    have shape := SharedReadback.single_shape r d (by simp [hp])
    simpa only [Shared.kernel,hp] using IOSafety.copy a (Shared.parameters r) t
      (sourceRow r j) (destinationRow r j) hs hd shape.1 shape.2 v bounds bs bd bk bv bkc bvc
  | int4 =>
    have sk : r.dk = 2*r.ck := by simpa [Request.factor,hp] using d.key_shape
    have sv : r.dv = 2*r.cv := by simpa [Request.factor,hp] using d.value_shape
    simpa only [Shared.kernel,hp] using IOSafety.quant4 a (Shared.parameters r) t
      (sourceRow r j) (destinationRow r j) hs hd pos.1 pos.2 widths.1 widths.2 sk sv v
      bounds bs bd bk bv bkc bvc (bkm (by simp [hp])) (bvm (by simp [hp]))
  | int8 =>
    have shape := SharedReadback.single_shape r d (by simp [hp])
    simpa only [Shared.kernel,hp] using IOSafety.quant8 a (Shared.parameters r) t
      (sourceRow r j) (destinationRow r j) hs hd pos.1 pos.2 widths.1 widths.2 shape.1 shape.2 v
      bounds bs bd bk bv bkc bvc (bkm (by simp [hp])) (bvm (by simp [hp]))

theorem interval_outside (base width offset : Nat)
    (h : ∀ k : Fin width, offset ≠ base+k.val) : offset < base ∨ base+width ≤ offset := by
  by_contra outside
  push Not at outside
  exact h ⟨offset-base,by omega⟩ (by dsimp; omega)

theorem outside_source {α : Type} (a : Ops α) (r : Request) (j : Job) (v : Bool) (pid : Nat)
    (vals : Inputs α r) (reg : RegionName) (i : Nat)
    (out : (programIO a r j v pid).Outside vals reg i) :
    SharedReadback.Outside r (destinationRow r j) reg i := by
  constructor
  · intro key
    by_cases hr : reg = output key
    · right
      subst reg
      cases key
      · apply interval_outside
        intro k
        have h := out.1 ⟨1,by simp [programIO,memory,outputs]⟩ (k,PUnit.unit) True.intro
        exact h.resolve_left (by simp [programIO,memory,outputs,output])
      · apply interval_outside
        intro k
        have h := out.1 ⟨0,by simp [programIO,memory,outputs]⟩ (k,PUnit.unit) True.intro
        exact h.resolve_left (by simp [programIO,memory,outputs,output])
    · exact Or.inl hr
  · intro hp key
    by_cases hr : reg = metaRegion key
    · right
      subst reg
      cases key
      · apply interval_outside
        intro k
        have h := out.1 ⟨3,by simp [programIO,memory,outputs]⟩ (k,PUnit.unit) hp
        exact h.resolve_left (by simp [programIO,memory,outputs,metaRegion])
      · apply interval_outside
        intro k
        have h := out.1 ⟨2,by simp [programIO,memory,outputs]⟩ (k,PUnit.unit) hp
        exact h.resolve_left (by simp [programIO,memory,outputs,metaRegion])
    · exact Or.inl hr

end
end VeriTile.Bench.Repairs.KvCacheFilling.Interface
