import bench.tritonbench_g.add_example.Proofs.FiniteSource

namespace VeriTile.Bench.TritonBenchG.AddExample.Finite
open VeriTile.Triton VeriTile.Bench.TritonBenchG.VectorAddition
open DTypeScope TypedHost VectorAddition.Finite
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def PhysicalView.Inside (v : PhysicalView) (address : Nat) : Prop :=
  v.base≤address ∧ address+bytes v.dtype≤v.base+v.capacity ∧ address+bytes v.dtype≤2^64

def PhysicalView.Disjoint (a b : PhysicalView) : Prop :=
  a.base+a.capacity≤b.base ∨ b.base+b.capacity≤a.base

def ReadOK {M A : Type} (ops : MemoryPrimitives M A) (v : PhysicalView) (address : Nat) : Prop :=
  v.Inside (v.toBacking address) ∧ address%bytes v.dtype=0 ∧
    v.deviceBase≤address ∧ address+bytes v.dtype≤v.deviceBase+v.capacity ∧ address+bytes v.dtype≤2^64 ∧
    ∀ k,k<bytes v.dtype → ops.readableByte ops.launchDevice (address+k) ∧
      v.toBacking (address+k)=v.toBacking address+k

def WriteOK {M A : Type} (ops : MemoryPrimitives M A) (v : PhysicalView) (address : Nat) : Prop :=
  v.Inside (v.toBacking address) ∧ address%bytes v.dtype=0 ∧
    v.deviceBase≤address ∧ address+bytes v.dtype≤v.deviceBase+v.capacity ∧ address+bytes v.dtype≤2^64 ∧
    ∀ k,k<bytes v.dtype → ops.writableByte ops.launchDevice (address+k) ∧
      v.toBacking (address+k)=v.toBacking address+k

theorem translated_load {M A : Type} (ops : MemoryPrimitives M A) (v : PhysicalView)
    (s : M) (native : Nat) (h : ReadOK ops v native) :
    ops.read s v.dtype (v.toBacking native)=
      ops.decode v.dtype (fun k => ops.observeByte s (v.toBacking (native+k))) := by
  rw [ops.read_bytes]
  apply ops.decode_local
  intro k hk
  rw [(h.2.2.2.2.2 k hk).2]

theorem translated_store_byte {M A : Type} (ops : MemoryPrimitives M A) (v : PhysicalView)
    (s : M) (native : Nat) (value : A) (h : WriteOK ops v native) (k : Nat) (hk : k<bytes v.dtype) :
    ops.observeByte (ops.write s v.dtype (v.toBacking native) value) (v.toBacking (native+k))=
      ops.encode v.dtype value k := by
  rw [(h.2.2.2.2.2 k hk).2]
  exact ops.store_byte s v.dtype (v.toBacking native) value k hk

/-- Complete externally observable frame, including typed reads and bytes. -/
def Preserved {M A : Type} (ops : MemoryPrimitives M A) (output : PhysicalView) (current initial : M) : Prop :=
  (∀ d address, SpanDisjoint output.base (output.base+output.capacity) address (address+bytes d) →
    ops.read current d address=ops.read initial d address) ∧
  (∀ address, address<output.base ∨ output.base+output.capacity≤address →
    ops.observeByte current address=ops.observeByte initial address)

theorem preserved_refl {M A : Type} (ops : MemoryPrimitives M A) (output : PhysicalView) (s : M) :
    Preserved ops output s s := ⟨fun _ _ _ => rfl,fun _ _ => rfl⟩

theorem preserved_trans {M A : Type} (ops : MemoryPrimitives M A) (output : PhysicalView) (a b c : M)
    (h : Preserved ops output a b) (k : Preserved ops output b c) : Preserved ops output a c :=
  ⟨fun d address outside => (h.1 d address outside).trans (k.1 d address outside),
   fun address outside => (h.2 address outside).trans (k.2 address outside)⟩

theorem write_preserved {M A : Type} (ops : MemoryPrimitives M A) (output : PhysicalView)
    (s : M) (w : Write A) (inside : output.Inside w.address) :
    Preserved ops output (ops.write s output.dtype w.address w.value) s := by
  constructor
  · intro d address outside
    apply ops.read_disjoint
    unfold SpanDisjoint at *
    unfold PhysicalView.Inside at inside
    rcases outside with h|h <;> omega
  · intro address outside
    apply ops.byte_frame
    unfold PhysicalView.Inside at inside
    rcases outside with h|h <;> omega

theorem writes_preserved {M A : Type} (ops : MemoryPrimitives M A) (output : PhysicalView)
    (writes : List (Write A)) (s : M) (inside : ∀ w∈writes,output.Inside w.address) :
    Preserved ops output (applyWrites ops output.dtype writes s) s := by
  induction writes generalizing s with
  | nil => exact preserved_refl ops output s
  | cons w rest ih =>
    exact preserved_trans ops output _ _ _
      (ih _ (fun w hw => inside w (by simp [hw]))) (write_preserved ops output s w (inside w (by simp)))

theorem input_preserved {M A : Type} (ops : MemoryPrimitives M A) (input output : PhysicalView)
    (current initial : M) (same : Preserved ops output current initial)
    (sep : input.Disjoint output) (address : Nat) (inside : input.Inside address) :
    ops.read current input.dtype address=ops.read initial input.dtype address := by
  apply same.1
  unfold PhysicalView.Disjoint at sep
  unfold PhysicalView.Inside at inside
  unfold SpanDisjoint
  rcases sep with h|h <;> omega

/-- Independent result at every byte, using the ordinary pure dtype packing
primitive. This remains meaningful for arbitrary tagged carrier values. -/
def byteFormula {M A : Type} (ops : MemoryPrimitives M A) (d : DType)
    (writes : List (Write A)) (initial : M) (address : Nat) : Fin 256 :=
  match writes.reverse.find? (fun w => decide (w.address≤address ∧ address<w.address+bytes d)) with
  | none => ops.observeByte initial address
  | some w => ops.encode d w.value (address-w.address)

theorem writes_append {M A : Type} (ops : MemoryPrimitives M A) (d : DType)
    (left right : List (Write A)) (s : M) :
    applyWrites ops d (left++right) s=applyWrites ops d right (applyWrites ops d left s) := by
  induction left generalizing s with
  | nil => rfl
  | cons w rest ih => exact ih _

theorem writes_pointwise {M A : Type} (ops : MemoryPrimitives M A) (d : DType)
    (writes : List (Write A)) (initial : M) (address : Nat) :
    ops.observeByte (applyWrites ops d writes initial) address=byteFormula ops d writes initial address := by
  induction writes using List.reverseRecOn with
  | nil => rfl
  | append_singleton rest w ih =>
    rw [writes_append]
    simp only [applyWrites]
    unfold byteFormula
    simp only [List.reverse_append,List.reverse_cons,List.reverse_nil,List.nil_append,List.cons_append,List.find?_cons]
    by_cases h : w.address≤address ∧ address<w.address+bytes d
    · have byte := ops.store_byte (applyWrites ops d rest initial) d w.address w.value (address-w.address) (by omega)
      have eq : w.address+(address-w.address)=address := by omega
      simp only [eq] at byte
      simp only [h,decide_true,if_true]
      exact byte
    · rw [ops.byte_frame _ _ _ _ _ (by omega)]
      simp [h,ih,byteFormula]

theorem writes_byte_frame {M A : Type} (ops : MemoryPrimitives M A) (d : DType)
    (writes : List (Write A)) (initial : M) (address : Nat)
    (outside : ∀ w∈writes,address<w.address ∨ w.address+bytes d≤address) :
    ops.observeByte (applyWrites ops d writes initial) address=ops.observeByte initial address := by
  induction writes generalizing initial with
  | nil => rfl
  | cons w rest ih =>
    rw [applyWrites,ih _ (fun v hv => outside v (by simp [hv]))]
    exact ops.byte_frame _ _ _ _ _ (outside w (by simp))

#specNonCircular byteFormula avoiding [applyWrites,runProgram,executeBody,executeInstruction]
#axiomsClean writes_preserved
#axiomsClean translated_load
#axiomsClean translated_store_byte
#axiomsClean input_preserved
#axiomsClean writes_pointwise
#axiomsClean writes_byte_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AddExample.Finite
