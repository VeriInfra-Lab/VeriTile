import bench.tritonbench_g.fused_rotary_embedding.Proofs.Physical

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding.Physical
open Native

/-- Metadata occupies its own actual four-byte int32 cells. Admitted positive
lengths and selected block IDs have bit31 clear; unused table entries are not
restricted. -/
def metadataWords (m : Memory Word) : Metadata :=
  ⟨fun t => (m .lengths t).toNat,fun a => (m .table a).toNat⟩
def metadata (m : Bytes) := metadataWords (load m)
def configuration (s : Shape) (m : Bytes) : Config := ⟨s,metadata m⟩

theorem signed32_value (w : Word) (h : w.toNat<2^31) : w.toInt=(w.toNat : Int) :=
  BitVec.toInt_eq_toNat_of_lt (by omega)
theorem admitted_length_decode (s : Shape) (m : Bytes) (hc : (configuration s m).Valid)
    (t : Nat) (ht : t<s.tokens) :
    (load m .lengths t).toInt=((configuration s m).metadata.lengths t : Int) :=
  signed32_value _ (hc.length32 t ht)
theorem admitted_block_decode (s : Shape) (m : Bytes) (hc : (configuration s m).Valid)
    (t : Nat) (ht : t<s.tokens) :
    (load m .table (t*s.tableWidth+(configuration s m).past t/s.blockSize)).toInt=
      ((configuration s m).block t : Int) :=
  signed32_value _ (hc.block32 t ht)
theorem signed_past (s : Shape) (m : Bytes) (hc : (configuration s m).Valid)
    (t : Nat) (ht : t<s.tokens) :
    (load m .lengths t).toInt-1=((configuration s m).past t : Int) := by
  rw [admitted_length_decode s m hc t ht]
  have positive := hc.lengthPositive t ht
  unfold Config.past
  omega

theorem metadata_word_store (m : Memory Word) (kind : Kind) (a : Nat) (v : Word) :
    metadataWords (Stores.store m ((kind.output,a),v))=metadataWords m := by
  cases kind <;> simp [metadataWords,Stores.store,Kind.output]
theorem metadata_byte_store (m : Bytes) (kind : Kind) (a : Nat) (v : Word) :
    metadata (store m kind.output a v)=metadata m := by
  unfold metadata
  rw [load_store]
  exact metadata_word_store (load m) kind a v
theorem byteStep_metadata (c : Config) (p : Instructions) (w : Work) (m : Bytes) :
    metadata (byteStep c p w m)=metadata m := by
  unfold byteStep
  rw [metadata_byte_store,metadata_byte_store]

/-- The source reads metadata from current bytes each time a KV work item
requires its destination. No destination or output is supplied by a callback. -/
def dynamicStep (s : Shape) (p : Instructions) (w : Work) (m : Bytes) :=
  byteStep (configuration s m) p w m
def dynamicExecute (s : Shape) (p : Instructions) : List Work → Bytes → Bytes
  | [],m => m
  | w::ws,m => dynamicExecute s p ws (dynamicStep s p w m)
def dynamic (s : Shape) (p : Instructions) (m : Bytes) := dynamicExecute s p (schedule s) m

theorem dynamicExecute_eq (c : Config) (p : Instructions) (ws : List Work) (m : Bytes)
    (hm : metadata m=c.metadata) : dynamicExecute c.shape p ws m=byteExecute c p ws m := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    have ce : configuration c.shape m=c := by cases c;simp_all [configuration]
    simp only [dynamicExecute,dynamicStep,ce,byteExecute]
    exact ih _ (by rw [byteStep_metadata];exact hm)
theorem dynamic_eq (s : Shape) (p : Instructions) (m : Bytes) :
    dynamic s p m=native (configuration s m) p m :=
  dynamicExecute_eq (configuration s m) p _ m rfl

theorem dynamic_source (s : Shape) (p : Instructions) (m : Bytes)
    (hc : (configuration s m).Valid) :
    load (dynamic s p m)=project (actualSource (configuration s m) (ops p) (liftMemory (load m))) := by
  rw [dynamic_eq]
  exact native_source (configuration s m) hc p m
theorem dynamic_result (s : Shape) (p : Instructions) (m : Bytes)
    (hc : (configuration s m).Valid) (w : Work) (hw : s.Active w) (o : Bool) :
    load (dynamic s p m) w.kind.output ((configuration s m).output w o)=pairWord s p w (load m) o := by
  rw [dynamic_eq]
  exact native_result (configuration s m) hc p m w hw o
theorem dynamic_bytes (s : Shape) (p : Instructions) (m : Bytes)
    (hc : (configuration s m).Valid) (w : Work) (hw : s.Active w) (o : Bool) (i : Fin 4) :
    dynamic s p m w.kind.output (4*(configuration s m).output w o+i.val)=encode (pairWord s p w (load m) o) i := by
  rw [dynamic_eq]
  exact native_bytes (configuration s m) hc p m w hw o i
theorem dynamic_frame (s : Shape) (p : Instructions) (m : Bytes) (r : Region) (b : Nat)
    (outside : ∀ w,s.Active w → ¬Writes (configuration s m) w r (b/4)) : dynamic s p m r b=m r b := by
  rw [dynamic_eq]
  exact native_frame (configuration s m) p m r b outside

theorem dynamic_metadata_frame (s : Shape) (p : Instructions) (m : Bytes) (table : Bool) (b : Nat) :
    dynamic s p m (if table then .table else .lengths) b=m (if table then .table else .lengths) b := by
  apply dynamic_frame
  intro w _
  cases table <;> cases kw : w.kind <;> simp [Writes,kw,Kind.output]

#axiomsClean dynamic_source
#axiomsClean dynamic_metadata_frame
end VeriTile.Bench.Repairs.FusedRotaryEmbedding.Physical
