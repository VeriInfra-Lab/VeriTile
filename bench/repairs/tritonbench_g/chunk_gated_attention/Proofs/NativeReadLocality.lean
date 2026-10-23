import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeStores

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeStores
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open NativeValue
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Exactly the finite typed reads reached by a recurrent source instance.
History and final output cells may differ arbitrarily between the two states. -/
structure InnerReadAgreement (c : NativeValue.Shape) (base : Nat) (s t : State NativeGraph.Expr)
    (gateK hasSeed : Bool) : Prop where
  pid : s.pids 0=t.pids 0
  key : ∀ chunk, chunk<c.NT → ∀ i : KIndex c,
    (s.mem "k" (keyOffset c base s chunk i)).read .real =
      (t.mem "k" (keyOffset c base t chunk i)).read .real
  value : ∀ chunk, chunk<c.NT → ∀ i : VIndex c,
    (s.mem "v" (valueOffset c base s chunk i)).read .real =
      (t.mem "v" (valueOffset c base t chunk i)).read .real
  gateKey : gateK=Bool.true → ∀ chunk, chunk<c.NT → ∀ i : KIndex c,
    (s.mem "g" (keyOffset c base s chunk i)).read .real =
      (t.mem "g" (keyOffset c base t chunk i)).read .real
  gateValue : gateK=Bool.false → ∀ chunk, chunk<c.NT → ∀ i : VIndex c,
    (s.mem "g" (valueOffset c base s chunk i)).read .real =
      (t.mem "g" (valueOffset c base t chunk i)).read .real
  keyEnd : gateK=Bool.true → ∀ chunk, chunk<c.NT → ∀ i : Fin c.BK,
    (s.mem "g" (keyEndpoint c base s chunk i)).read .real =
      (t.mem "g" (keyEndpoint c base t chunk i)).read .real
  valueEnd : gateK=Bool.false → ∀ chunk, chunk<c.NT → ∀ i : Fin c.BV,
    (s.mem "g" (valueEndpoint c base s chunk i)).read .real =
      (t.mem "g" (valueEndpoint c base t chunk i)).read .real
  seed : hasSeed=Bool.true → ∀ i : HIndex c,
    (s.mem "h0" (stateOffset c base s i)).read .fp32 =
      (t.mem "h0" (stateOffset c base t i)).read .fp32

theorem initial_eq (c : NativeValue.Shape) (base : Nat) (s t : State NativeGraph.Expr)
    (gateK hasSeed : Bool) (agree : InnerReadAgreement c base s t gateK hasSeed) :
    NativeValue.initial algebra c base s hasSeed = NativeValue.initial algebra c base t hasSeed := by
  funext i
  cases hasSeed
  · rfl
  · simp only [NativeValue.initial,↓reduceIte,agree.seed rfl i]

theorem key_eq (c : NativeValue.Shape) (base : Nat) (s t : State NativeGraph.Expr)
    (gateK hasSeed : Bool) (agree : InnerReadAgreement c base s t gateK hasSeed)
    (chunk : Nat) (active : chunk<c.NT) :
    keyValue algebra c base s gateK chunk=keyValue algebra c base t gateK chunk := by
  funext i
  cases gateK
  · simp only [keyValue,load,Bool.false_eq_true,↓reduceIte,agree.key chunk active i]
  · simp only [keyValue,load,↓reduceIte,agree.key chunk active i,
      agree.gateKey rfl chunk active i,agree.keyEnd rfl chunk active i.1]

theorem value_eq (c : NativeValue.Shape) (base : Nat) (s t : State NativeGraph.Expr)
    (gateK hasSeed : Bool) (agree : InnerReadAgreement c base s t gateK hasSeed)
    (chunk : Nat) (active : chunk<c.NT) :
    valueValue algebra c base s gateK chunk=valueValue algebra c base t gateK chunk := by
  funext i
  cases gateK
  · simp only [valueValue,load,Bool.false_eq_true,↓reduceIte,agree.value chunk active i,
      agree.gateValue rfl chunk active i,agree.valueEnd rfl chunk active i.2.1]
  · simp only [valueValue,load,↓reduceIte,agree.value chunk active i]

theorem gated_eq (c : NativeValue.Shape) (base : Nat) (s t : State NativeGraph.Expr)
    (gateK hasSeed : Bool) (agree : InnerReadAgreement c base s t gateK hasSeed)
    (chunk : Nat) (active : chunk<c.NT) (acc : HIndex c → NativeGraph.Expr) :
    gated algebra c base s gateK chunk acc=gated algebra c base t gateK chunk acc := by
  funext i
  cases gateK
  · simp only [gated,load,Bool.false_eq_true,↓reduceIte,agree.valueEnd rfl chunk active i.2.1]
  · simp only [gated,load,↓reduceIte,agree.keyEnd rfl chunk active i.1]

theorem inner_step_eq (c : NativeValue.Shape) (base : Nat) (s t : State NativeGraph.Expr)
    (gateK hasSeed : Bool) (agree : InnerReadAgreement c base s t gateK hasSeed)
    (chunk : Nat) (active : chunk<c.NT) (acc : Accumulator c) :
    innerStep c base s gateK chunk acc=innerStep c base t gateK chunk acc := by
  have ku := key_eq c base s t gateK hasSeed agree chunk active
  have vu := value_eq c base s t gateK hasSeed agree chunk active
  have gu := gated_eq c base s t gateK hasSeed agree chunk active acc.1
  apply Prod.ext
  · change NativeValue.update c base s gateK chunk acc.1 = NativeValue.update c base t gateK chunk acc.1
    funext i
    unfold NativeValue.update
    rw [ku,vu,congrFun gu i]
  · change acc.2++history c base s chunk acc.1 = acc.2++history c base t chunk acc.1
    simp only [history,historyOffset,head,kblock,vblock,linear,agree.pid]

theorem inner_range_eq (c : NativeValue.Shape) (base : Nat) (s t : State NativeGraph.Expr)
    (gateK hasSeed : Bool) (agree : InnerReadAgreement c base s t gateK hasSeed)
    (cur : Nat) (acc : Accumulator c) :
    foldRange (innerStep c base s gateK) cur c.NT 1 acc =
      foldRange (innerStep c base t gateK) cur c.NT 1 acc := by
  conv_lhs => rw [foldRange]
  conv_rhs => rw [foldRange]
  simp only [one_ne_zero,↓reduceDIte]
  split_ifs with active
  · rw [inner_step_eq c base s t gateK hasSeed agree cur active acc]
    exact inner_range_eq c base s t gateK hasSeed agree (cur+1) _
  · rfl
termination_by c.NT-cur
decreasing_by omega

/-- Equal source reads imply the identical complete list of history and final
native stores. No initial output value, register, or undef equality is needed. -/
theorem inner_eq (c : NativeValue.Shape) (base : Nat) (s t : State NativeGraph.Expr)
    (gateK hasSeed storeFinal : Bool) (agree : InnerReadAgreement c base s t gateK hasSeed) :
    inner c base s gateK hasSeed storeFinal=inner c base t gateK hasSeed storeFinal := by
  simp only [inner,initial_eq c base s t gateK hasSeed agree,
    inner_range_eq c base s t gateK hasSeed agree]
  simp only [final,stateOffset,head,kblock,vblock,linear,agree.pid]

structure PrefixReadAgreement (c : NativePrefix.Shape) (base : Nat) (s t : State NativeGraph.Expr) : Prop where
  pid : s.pids 0=t.pids 0
  input : NativePrefix.enabled c base s → ∀ i : NativePrefix.Index c,
    (s.mem "s" (NativePrefix.offset c base s i)).read .fp32 =
      (t.mem "s" (NativePrefix.offset c base t i)).read .fp32

theorem prefix_eq (c : NativePrefix.Shape) (base : Nat) (s t : State NativeGraph.Expr)
    (agree : PrefixReadAgreement c base s t) : prefixStores c base s=prefixStores c base t := by
  have enabled : NativePrefix.enabled c base s ↔ NativePrefix.enabled c base t := by
    simp only [NativePrefix.enabled,NativePrefix.linear,agree.pid]
  by_cases active : NativePrefix.enabled c base s
  · have inputs : NativePrefix.input c base s=NativePrefix.input c base t := by
      funext i
      simp only [NativePrefix.input,agree.input active i]
    simp only [prefixStores,if_pos active,if_pos (enabled.mp active),NativePrefix.value,inputs]
    simp only [NativePrefix.offset,NativePrefix.head,NativePrefix.timeBlock,NativePrefix.featureBlock,
      NativePrefix.linear,agree.pid]
  · simp only [prefixStores,if_neg active,if_neg (fun ht => active (enabled.mpr ht))]

#axiomsClean inner_eq
#axiomsClean prefix_eq
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeStores
