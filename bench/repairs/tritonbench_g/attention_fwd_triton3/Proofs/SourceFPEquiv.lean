import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.HostRetry
import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.CallMachine

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.FPEquiv
open VeriTile.Triton Host Ownership Raw NativeSchedule
noncomputable section

/-- The complete native reference computes every O/M/L word from six initial
arrays, physical reduction trees and fixed K=8 instruction chains. Only finite
writer indices are chosen. This is independent of the actual final bytes. -/
def reference (c : Call) (k : Kind) (p : Profile c k) (choice : Selection c)
    (ops : Ops) (A : FlatAlloc) (before : Physical.Bytes) (scale : Word) : Physical.Bytes :=
  Physical.execute c A ops (schedule c k p choice) before scale

/-- Full bytes, exact frame and public returned shape/identities for each of
the six pinned native profiles. The saved compiler replay establishes the
profile's layout/routes; no universal compiler-correctness claim is made. -/
theorem whole_bytes (c : Call) (h : Valid c) (A : FlatAlloc) (a : Allocated c A.extent)
    (hd : A.Disjoint) (hc : ∀ rg,rg∉A.regions → A.extent rg=0)
    (k : Kind) (p : Profile c k) (choice : Selection c) (ops : Ops)
    (before : Physical.Bytes) (scale : Word) :
    (∀ idx i b,active c i → reference c k p choice ops A before scale (Physical.address c A idx i b)=
      RawStores.byte (output c ops (schedule c k p choice) (Physical.snapshot A before) scale idx i) b) ∧
    (∀ addr,(¬∃ idx i b,active c i ∧ Physical.address c A idx i b=addr) →
      reference c k p choice ops A before scale addr=before addr) ∧
    (Physical.returned c A (reference c k p choice ops A before scale)).shape=
      [c.Z,c.H,if c.finish=0 then c.rounded else c.NQ,c.D] ∧
    (Physical.returned c A (reference c k p choice ops A before scale)).sameM=Bool.true ∧
    (Physical.returned c A (reference c k p choice ops A before scale)).sameL=Bool.true ∧
    (c.finish=0 → (Physical.returned c A (reference c k p choice ops A before scale)).guaranteesSameO=Bool.true) := by
  refine ⟨?_,?_,rfl,rfl,rfl,?_⟩
  · exact Physical.readback c h A a hd hc ops (schedule c k p choice) before scale
  · exact Physical.frame c A ops (schedule c k p choice) before scale
  · intro he
    simp [Physical.returned,he]

/-- Nine genuinely local numerical families: eight scalar instructions/fixed
unary helpers and one coordinate of a fixed K=8 matrix instruction. Every
finite writer selection is covered; no commutativity, associativity, whole-row
identity, whole-dot callback or whole-output equality is a premise. -/
theorem whole_equiv (c : Call) (A : FlatAlloc) (k : Kind) (p : Profile c k)
    (choice : Selection c) (actual expected : Ops) (laws : LocalLaws actual expected)
    (before : Physical.Bytes) (scale : Word) :
    reference c k p choice actual A before scale=reference c k p choice expected A before scale ∧
    Physical.returned c A (reference c k p choice actual A before scale)=
      Physical.returned c A (reference c k p choice expected A before scale) :=
  Physical.whole_equiv c A actual expected laws (schedule c k p choice) before scale

/-- The generic full graph also covers unlisted branch combinations under
an ordinary compiler schedule contract. It has no output-valued schedule
field; replay evidence for concrete compilers is separately scoped above. -/
theorem graph_equiv (c : Call) (actual expected : Ops) (laws : LocalLaws actual expected)
    (s : Schedule c) (memory : Memory) (scale : Word) :
    output c actual s memory scale=output c expected s memory scale ∧
    RawStores.bytes c actual s memory scale=RawStores.bytes c expected s memory scale :=
  ⟨complete_equiv c actual expected laws s memory scale,
   RawStores.whole_bytes c actual expected laws s memory scale⟩

#print axioms whole_bytes
#print axioms whole_equiv
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.FPEquiv
