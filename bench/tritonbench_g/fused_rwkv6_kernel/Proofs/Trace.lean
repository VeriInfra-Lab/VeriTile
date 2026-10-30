import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.Source
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 10000000

/-- Finite actual masked-store locations in program/time/lane order. -/
def partialTrace (c : Shape) (p : Program) : List (Region × Nat) :=
  (List.range c.T).flatMap fun n => ((List.range c.BV).filter fun l => valueIndex c p l<c.V).map
    (fun l => (.o,partialAddress c p.z p.ik (visit c n) (valueIndex c p l)))

def finalTrace (c : Shape) (p : Program) : List (Region × Nat) :=
  if c.final then (List.range c.BV).flatMap fun l =>
    ((List.range c.BK).filter fun j => valueIndex c p l<c.V ∧ keyIndex c p j<c.K).map
      (fun j => (.ht,stateAddress c p.z (keyIndex c p j) (valueIndex c p l))) else []

def wordTrace (c : Shape) : List (Region × Nat) :=
  (programs c).flatMap (fun p => partialTrace c p ++ finalTrace c p) ++
    (List.range (c.Z*c.T*c.V)).map (fun b => (.result,b))

theorem partial_trace_mem (c : Shape) (p : Program) (r : Region) (b : Nat) :
    (r,b)∈partialTrace c p ↔ ∃ n<c.T,PartialWrite c p n r b := by
  simp only [partialTrace,List.mem_flatMap,List.mem_map,List.mem_filter,List.mem_range,
    decide_eq_true_eq,Prod.mk.injEq,PartialWrite]
  aesop

theorem final_trace_mem (c : Shape) (p : Program) (r : Region) (b : Nat) :
    (r,b)∈finalTrace c p ↔ FinalWrite c p r b := by
  cases hs : c.final <;>
    simp only [finalTrace,hs,Bool.false_eq_true,if_false,List.not_mem_nil,FinalWrite,
      false_and,Bool.true_eq,true_and,if_true,List.mem_flatMap,List.mem_map,List.mem_filter,
      List.mem_range,decide_eq_true_eq,Prod.mk.injEq]
  aesop

theorem word_trace_mem (c : Shape) (hc : Sizes c) (r : Region) (b : Nat) :
    (r,b)∈wordTrace c ↔ WholeOutput c r b := by
  simp only [wordTrace,List.mem_append,List.mem_flatMap,partial_trace_mem,final_trace_mem,
    List.mem_map,List.mem_range,Prod.mk.injEq]
  constructor
  · rintro (⟨p,hp,hw⟩|⟨d,hd,hr,hb⟩)
    · exact Or.inl (writes_output c p ((mem_programs c p).1 hp) hw)
    · exact Or.inr ⟨hr.symm,hb ▸ hd⟩
  · rintro (h|⟨rfl,h⟩)
    · exact Or.inl (cover c hc h)
    · exact Or.inr ⟨b,h,rfl,rfl⟩

structure ByteEvent where
  address : Nat
  value : BitVec 8

def byteTrace (c : Shape) (layout : Layout c) (words : Memory (BitVec 32)) : List ByteEvent :=
  (wordTrace c).flatMap fun (r,b) => List.ofFn fun j : Fin 4 =>
    ⟨layout.base r+4*b+j.val,byteOf (words r b) j⟩

def writeByte (m : Bytes) (e : ByteEvent) : Bytes := fun addr =>
  if addr=e.address then e.value else m addr

def executeBytes (events : List ByteEvent) (memory : Bytes) : Bytes := events.foldl writeByte memory

theorem execute_consistent (events : List ByteEvent) (post memory : Bytes)
    (h : ∀ e∈events,e.value=post e.address) (addr : Nat) :
    executeBytes events memory addr=
      if ∃ e∈events,e.address=addr then post addr else memory addr := by
  induction events generalizing memory with
  | nil => simp [executeBytes]
  | cons e es ih =>
    change executeBytes es (writeByte memory e) addr = _
    rw [ih _ (fun e he => h e (by simp [he]))]
    have he := h e (by simp)
    by_cases tail : ∃ e∈es,e.address=addr
    · simp [tail]
    · by_cases hit : e.address=addr
      · simp [tail,hit,writeByte,he]
      · simp [tail,hit,writeByte,Ne.symm hit]

theorem byte_trace_mem (c : Shape) (layout : Layout c) (words : Memory (BitVec 32)) (e : ByteEvent) :
    e∈byteTrace c layout words ↔ ∃ r b, (r,b)∈wordTrace c ∧ ∃ j : Fin 4,
      e=⟨layout.base r+4*b+j.val,byteOf (words r b) j⟩ := by
  simp only [byteTrace,List.mem_flatMap,List.mem_ofFn]
  constructor
  · rintro ⟨⟨r,b⟩,h,j,hj⟩
    exact ⟨r,b,h,j,hj.symm⟩
  · rintro ⟨r,b,h,j,rfl⟩
    exact ⟨(r,b),h,j,rfl⟩

/-- Explicit finite byte stores realize the native memory result; arbitrary
initial bytes are permitted and every outside byte is preserved. -/
theorem byte_trace_realizes (c : Shape) (hc : Sizes c) (layout : Layout c) (words : Memory (BitVec 32))
    (memory : Bytes) : executeBytes (byteTrace c layout words) memory=scatterBytes c layout memory words := by
  funext addr
  have h : ∀ e∈byteTrace c layout words,e.value=scatterBytes c layout memory words e.address := by
    intro e he
    obtain ⟨r,b,hw,j,rfl⟩ := (byte_trace_mem c layout words e).1 he
    exact (scatter_byte c layout memory words r b j ((word_trace_mem c hc r b).1 hw)).symm
  rw [execute_consistent _ _ _ h]
  by_cases he : ∃ e∈byteTrace c layout words,e.address=addr
  · simp [he]
  · rw [if_neg he]
    symm
    apply scatter_frame
    intro r b j hd hw
    have hb := (word_trace_mem c hc r b).2 hw
    have ha := (layout.sound addr r b j hd).2
    apply he
    refine ⟨⟨layout.base r+4*b+j.val,byteOf (words r b) j⟩,?_,ha.symm⟩
    exact (byte_trace_mem c layout words _).2 ⟨r,b,hb,j,rfl⟩

theorem native_trace (a : Native.Instructions) (compiler : Native.Compiler) (c : Shape)
    (hc : Sizes c) (plan : Plans c) (scale : BitVec 32) (layout : Layout c) (memory : Bytes) :
    executeBytes (byteTrace c layout
      (Native.nativeForward a compiler c plan scale (loadWords c layout memory))) memory=
      physicalForward a compiler c plan scale layout memory :=
  byte_trace_realizes c hc layout _ memory

#axiomsClean word_trace_mem
#axiomsClean byte_trace_realizes
#axiomsClean native_trace
end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel
