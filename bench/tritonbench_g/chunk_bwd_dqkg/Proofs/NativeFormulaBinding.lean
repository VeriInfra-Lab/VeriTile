import bench.tritonbench_g.chunk_bwd_dqkg.Target
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.AddressBounds

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.NativeFormulaBinding
open VeriTile Triton FP.Structural Whole KernelNativeCarrier AddressBounds
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

/-- The source's two interpreter channels are decoded exactly as before. -/
def read (s : State Node) (r : RegionName) (i : Nat) : Node :=
  if r="q" ∨ r="k" ∨ r="v" ∨ r="do" then (s.mem r i).read .fp32 else (s.mem r i).read .real
def ReadAgreement (c : Config) (s : State Node) (μ : RegionName → Nat → Node) : Prop :=
  ∀ r,r∈(["q","k","v","h","g","do","dh"] : List RegionName) →
    ∀ i,i<extent c r → read s r i=μ r i

theorem logical_inputs (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (order : List (CTA c)) (s : State Node) (vals : ChannelValues Node (IO.nativeInputs c))
    (hin : (IO.nativeIO isa route scale c order).InputBinding vals s) :
    ReadAgreement c s (Target.inputs c (IO.nativeValues c vals)) := by
  intro r hr n hn
  simp only [List.mem_cons,List.mem_nil_iff,or_false] at hr
  rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp only [read,Target.inputs,ite_true,ite_false,reduceCtorEq,or_self,or_false,false_or]
  · simpa [Target.get,IO.nativeValues,IO.inputSize,IO.inputRegion,hn] using hin (0 : Fin 7) (⟨n,hn⟩,PUnit.unit) trivial
  · simpa [Target.get,IO.nativeValues,IO.inputSize,IO.inputRegion,hn] using hin (1 : Fin 7) (⟨n,hn⟩,PUnit.unit) trivial
  · simpa [Target.get,IO.nativeValues,IO.inputSize,IO.inputRegion,hn] using hin (2 : Fin 7) (⟨n,hn⟩,PUnit.unit) trivial
  · simpa [Target.get,IO.nativeValues,IO.inputSize,IO.inputRegion,hn] using hin (3 : Fin 7) (⟨n,hn⟩,PUnit.unit) trivial
  · simpa [Target.get,IO.nativeValues,IO.inputSize,IO.inputRegion,hn] using hin (4 : Fin 7) (⟨n,hn⟩,PUnit.unit) trivial
  · simpa [Target.get,IO.nativeValues,IO.inputSize,IO.inputRegion,hn] using hin (5 : Fin 7) (⟨n,hn⟩,PUnit.unit) trivial
  · simpa [Target.get,IO.nativeValues,IO.inputSize,IO.inputRegion,hn] using hin (6 : Fin 7) (⟨n,hn⟩,PUnit.unit) trivial

theorem primitives (isa : Instructions) (route : Routing) (c : Config) (s : State Node)
    (μ : RegionName → Nat → Node) (reads : ReadAgreement c s μ) (a : CTA c) :
    KernelNativeReference.gate isa route c (KernelNative.entry c s a)=Target.Native.gate isa route c μ a ∧
    KernelNativeReference.lastGate isa route c (KernelNative.entry c s a)=Target.Native.lastGate isa route c μ a ∧
    (∀ (r : RegionName),r="v" ∨ r="do" → ∀ block,
      KernelNativeReference.value c (KernelNative.entry c s a) r block=Target.Native.value c μ a r block) ∧
    (∀ (r : RegionName),r="h" ∨ r="dh" → ∀ block,
      KernelNativeReference.history isa route c (KernelNative.entry c s a) r block=Target.Native.history isa route c μ a r block) ∧
    (∀ (r : RegionName),r="q" ∨ r="k" →
      KernelNativeReference.queryKey c (KernelNative.entry c s a) r=Target.Native.queryKey c μ a r) := by
  have cap : Capacity c (extent c):=fun _=>le_rfl
  refine ⟨?_,?_,?_,?_,?_⟩
  · funext r
    simp only [KernelNativeReference.gate,Target.Native.gate,KernelNative.entry]
    norm_num only
    simp only [ite_false,ite_true]
    congr 1
    split_ifs with active
    · simpa [read,Nat.add_assoc] using reads "g" (by simp) _
        (gate_bound c a (extent c) cap _ active)
    · rfl
  · simp only [KernelNativeReference.lastGate,Target.Native.lastGate,KernelNative.entry]
    norm_num only
    simp only [ite_false,ite_true]
    congr 1
    have hl : min (a.time.val*64+64) c.T-1<c.T := by have:=T_pos c a;omega
    simpa [read] using reads "g" (by simp) _ (gate_bound c a (extent c) cap _ hl)
  · intro r hr block
    funext i
    simp only [KernelNativeReference.value,Target.Native.value,KernelNative.entry]
    norm_num only
    simp only [ite_false,ite_true]
    split_ifs with active
    · have h:=reads r (by rcases hr with rfl|rfl <;> simp) _
        (value_bound c a (extent c) cap r hr _ _ active.1 active.2)
      rcases hr with rfl|rfl <;> simpa [read] using h
    · rfl
  · intro r hr block
    funext i
    simp only [KernelNativeReference.history,Target.Native.history,KernelNative.entry]
    norm_num only
    simp only [ite_false,ite_true]
    congr 1
    split_ifs with active
    · have h:=reads r (by rcases hr with rfl|rfl <;> simp) _
        (history_bound c a (extent c) cap r hr _ _ active.1 active.2)
      rcases hr with rfl|rfl <;> simpa [read] using h
    · rfl
  · intro r hr
    funext i
    simp only [KernelNativeReference.queryKey,Target.Native.queryKey,KernelNative.entry]
    norm_num only
    simp only [ite_false,ite_true]
    split_ifs with active
    · have h:=reads r (by rcases hr with rfl|rfl <;> simp) _
        (query_bound c a (extent c) cap r (by tauto) _ _ active.1 active.2)
      rcases hr with rfl|rfl <;> simpa [read] using h
    · rfl

theorem accumulate (isa : Instructions) (route : Routing) (c : Config) (s : State Node)
    (μ : RegionName → Nat → Node) (reads : ReadAgreement c s μ) (a : CTA c) (n : Nat) :
    KernelNativeReference.accumulate isa route c (KernelNative.entry c s a) n=
      Target.Native.accumulate isa route c μ a n := by
  obtain ⟨_,_,hv,hh,_⟩:=primitives isa route c s μ reads a
  induction n with
  | zero=>rfl
  | succ n ih=>
    simp only [KernelNativeReference.accumulate,Target.Native.accumulate,ih,
      KernelNativeReference.step,Target.Native.step,hv "v" (Or.inl rfl),hv "do" (Or.inr rfl),
      hh "h" (Or.inl rfl),hh "dh" (Or.inr rfl)]

theorem answers (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node)
    (μ : RegionName → Nat → Node) (reads : ReadAgreement c s μ) (a : CTA c)
    (carry : KernelNativeReference.Carry c) :
    KernelNativeReference.query isa route scale c (KernelNative.entry c s a) carry=
      Target.Native.query isa route scale c μ a carry ∧
    KernelNativeReference.key isa route scale c (KernelNative.entry c s a) carry=
      Target.Native.key isa route scale c μ a carry ∧
    KernelNativeReference.gateResult isa route scale c (KernelNative.entry c s a) carry=
      Target.Native.gateResult isa route scale c μ a carry := by
  obtain ⟨hg,hl,_,_,hk⟩:=primitives isa route c s μ reads a
  have hq:=hk "q" (Or.inl rfl)
  have hk:=hk "k" (Or.inr rfl)
  have hkg : KernelNativeReference.gatedK isa route c (KernelNative.entry c s a) carry=
      Target.Native.gatedK isa route c μ a carry := by
    funext i
    simp only [KernelNativeReference.gatedK,hg,hl]
    rfl
  refine ⟨?_,?_,?_⟩
  all_goals funext i
  all_goals simp only [KernelNativeReference.query,KernelNativeReference.key,
    KernelNativeReference.gateResult,KernelNativeReference.gatedQ,KernelNativeReference.gatedK,
    KernelNativeReference.scores,KernelNativeReference.correction]
  all_goals simp only [hg,hl,hq,hk,hkg]
  all_goals rfl

end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.NativeFormulaBinding
