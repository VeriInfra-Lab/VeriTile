import bench.repairs.tritonbench_g.rotary_transform.Proofs.Physical
namespace VeriTile.Bench.Repairs.RotaryTransform.Physical
open Native
/-- CU may have positive element stride; offsets are contiguous. Int64 is
loaded as two little-endian words, before the checked source i32 conversion. -/
structure MetadataABI where
  cumulative64 : Bool
  offsets64 : Bool
  cumulativeStride : Nat
  deriving DecidableEq, Repr
def MetadataABI.unit (a : MetadataABI) : MetadataABI := { a with cumulativeStride := 1 }
def integer (m : Bytes) (r : Region) (wide : Bool) (index : Nat) : Nat :=
  if wide then ((load m r (2*index+1) ++ load m r (2*index)) : BitVec 64).toNat
  else (load m r index).toNat
def metadata (a : MetadataABI) (m : Bytes) : Metadata :=
  ⟨fun i => integer m .cumulative a.cumulative64 (i*a.cumulativeStride),
   fun i => integer m .offsets a.offsets64 i⟩
def originalMetadata (a : MetadataABI) (m : Bytes) : Metadata :=
  ⟨fun i => integer m (.other 0) a.cumulative64 (i*a.cumulativeStride),
   fun i => integer m .offsets a.offsets64 i⟩
def canonicalMetadata (s : Shape) (md : Metadata) : Metadata :=
  ⟨if s.varlen then md.cumulative else fun _ => 0,
   if s.offsetTensor then md.offsets else fun _ => 0⟩
def originalConfiguration (s : Shape) (a : MetadataABI) (m : Bytes) : Config := ⟨s,canonicalMetadata s (originalMetadata a m)⟩
def configuration (s : Shape) (a : MetadataABI) (m : Bytes) : Config := ⟨s,canonicalMetadata s (metadata a m)⟩

/-- A private normalized CU allocation. Caller CU storage is other 0; the runtime cumulative region holds the contiguous copy.
This definition copies only input bytes, never an output formula. -/
def normalize (a : MetadataABI) (m : Bytes) : Bytes := fun r b =>
  if r=.cumulative then
    let n := if a.cumulative64 then 8 else 4
    m (.other 0) ((b/n)*a.cumulativeStride*n+b%n)
  else m r b

theorem normalize_other (a : MetadataABI) (m : Bytes) (r : Region) (b : Nat) (h : r≠.cumulative) :
    normalize a m r b=m r b := by simp [normalize,h]
theorem normalized_word32 (a : MetadataABI) (m : Bytes) (i : Nat) (h : a.cumulative64=false) :
    load (normalize a m) .cumulative i=load m (.other 0) (i*a.cumulativeStride) := by
  unfold load
  congr 1
  funext j
  have hd : (4*i+j.val)/4=i := by omega
  have hm : (4*i+j.val)%4=j.val := by omega
  simp only [normalize,h,Bool.false_eq_true,↓reduceIte,hd,hm]
  congr 1
  ring
theorem normalized_word64 (a : MetadataABI) (m : Bytes) (i : Nat) (o : Bool)
    (h : a.cumulative64=true) :
    load (normalize a m) .cumulative (2*i+if o then 1 else 0)=
      load m (.other 0) (2*(i*a.cumulativeStride)+if o then 1 else 0) := by
  unfold load
  congr 1
  funext j
  have hd : (4*(2*i+if o then 1 else 0)+j.val)/8=i := by cases o <;> simp <;> omega
  have hm : (4*(2*i+if o then 1 else 0)+j.val)%8=4*(if o then 1 else 0)+j.val := by
    cases o <;> simp <;> omega
  simp only [normalize,h,Bool.false_eq_true,↓reduceIte,hd,hm]
  congr 1
  ring
theorem normalize_metadata (a : MetadataABI) (m : Bytes) : metadata a.unit (normalize a m)=originalMetadata a m := by
  change Metadata.mk _ _=Metadata.mk _ _
  apply congrArg₂ Metadata.mk
  · funext i
    simp only [metadata,originalMetadata,MetadataABI.unit,Nat.mul_one,integer]
    cases h : a.cumulative64
    · simp only [h,Bool.false_eq_true,↓reduceIte]
      rw [normalized_word32 a m i h]
    · simp only [h,↓reduceIte]
      rw [show 2*i+1=2*i+(if true then 1 else 0) from rfl,
        normalized_word64 a m i true h,
        show 2*i=2*i+(if false then 1 else 0) from (Nat.add_zero _).symm,
        normalized_word64 a m i false h]
      rfl
  · funext i
    simp [metadata,originalMetadata,MetadataABI.unit,integer,load,normalize]

def normalizeFor (s : Shape) (a : MetadataABI) (m : Bytes) := if s.varlen then normalize a m else m
theorem normalizeFor_configuration (s : Shape) (a : MetadataABI) (m : Bytes) :
    configuration s a.unit (normalizeFor s a m)=originalConfiguration s a m := by
  cases h : s.varlen
  · simp [configuration,originalConfiguration,canonicalMetadata,normalizeFor,h,
      metadata,originalMetadata,MetadataABI.unit]
  · simp [configuration,originalConfiguration,normalizeFor,h,normalize_metadata]

theorem metadata_store (a : MetadataABI) (m : Bytes) (s : Shape) (i : Nat) (v : Word) :
    metadata a (store m s.output i v)=metadata a m := by
  cases h : s.inplace <;> simp [metadata,integer,load,store,Shape.output,h]
theorem byteStep_metadata (a : MetadataABI) (c : Config) (p : Instructions) (w : Work) (m : Bytes) :
    metadata a (byteStep c p w m)=metadata a m := by
  unfold byteStep
  rw [metadata_store,metadata_store]

def dynamicStep (s : Shape) (a : MetadataABI) (p : Instructions) (w : Work) (m : Bytes) :=
  byteStep (configuration s a m) p w m
def dynamicExecute (s : Shape) (a : MetadataABI) (p : Instructions) : List Work → Bytes → Bytes
  | [],m => m
  | w::ws,m => dynamicExecute s a p ws (dynamicStep s a p w m)
def dynamic (s : Shape) (a : MetadataABI) (p : Instructions) (m : Bytes) :=
  dynamicExecute s a p (schedule (configuration s a m)) m

theorem dynamicExecute_eq (c : Config) (a : MetadataABI) (p : Instructions) (ws : List Work) (m : Bytes)
    (hm : canonicalMetadata c.shape (metadata a m)=c.metadata) : dynamicExecute c.shape a p ws m=byteExecute c p ws m := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    have ce : configuration c.shape a m=c := by cases c;simp_all [configuration]
    simp only [dynamicExecute,dynamicStep,ce,byteExecute]
    exact ih _ (by rw [byteStep_metadata];exact hm)
theorem dynamic_eq (s : Shape) (a : MetadataABI) (p : Instructions) (m : Bytes) :
    dynamic s a p m=native (configuration s a m) p m :=
  dynamicExecute_eq (configuration s a m) a p _ m rfl

theorem signed32 (w : BitVec 32) (h : w.toNat<2^31) : w.toInt=(w.toNat : Int) :=
  BitVec.toInt_eq_toNat_of_lt (by omega)
theorem signed64 (w : BitVec 64) (h : w.toNat<2^31) : w.toInt=(w.toNat : Int) :=
  BitVec.toInt_eq_toNat_of_lt (by omega)
#specNonCircular metadata avoiding [source,pairValues,dynamic,native]
#axiomsClean normalize_metadata
#axiomsClean dynamic_eq
#axiomsClean signed32
#axiomsClean signed64
end VeriTile.Bench.Repairs.RotaryTransform.Physical
