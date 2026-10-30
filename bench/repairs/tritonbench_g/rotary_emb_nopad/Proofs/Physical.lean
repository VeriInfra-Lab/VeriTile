import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.Native
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Physical

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Physical
open Checked Native
abbrev Byte := VeriTile.Bench.Repairs.RotaryEmb.Physical.Byte
abbrev Bytes := VeriTile.Bench.Repairs.RotaryEmb.Physical.Bytes
abbrev load := VeriTile.Bench.Repairs.RotaryEmb.Physical.load
abbrev store := VeriTile.Bench.Repairs.RotaryEmb.Physical.store
abbrev encode := VeriTile.Bench.Repairs.RotaryEmb.Physical.encode

/-- Actual little-endian signed-int64 metadata, at a four-byte word address. -/
def indices (r : Request) (m : Bytes) : Indices := fun address =>
  ((load m (.other r.device) (address+1) ++ load m (.other r.device) address) : BitVec 64).toInt

def byteStep (p : Instructions) (r : Request) (index : Indices) (w : Work) (m : Bytes) : Bytes :=
  let low := Native.expected p r w (load m) false
  let high := Native.expected p r w (load m) true
  store (store m (.other r.device) (destination r index w false) low)
    (.other r.device) (destination r index w true) high

theorem step_load (p : Instructions) (r : Request) (index : Indices) (w : Work) (m : Bytes) :
    load (byteStep p r index w m)=wordStep p r index w (load m) := by
  simp only [byteStep,wordStep,load,store]
  rw [VeriTile.Bench.Repairs.RotaryEmb.Physical.load_store,VeriTile.Bench.Repairs.RotaryEmb.Physical.load_store]

def byteExecute (p : Instructions) (r : Request) (index : Indices) : List Work → Bytes → Bytes
  | [],m => m
  | w::rest,m => byteExecute p r index rest (byteStep p r index w m)

theorem execute_load (p : Instructions) (r : Request) (index : Indices) (jobs : List Work) (m : Bytes) :
    load (byteExecute p r index jobs m)=wordExecute p r index jobs (load m) := by
  induction jobs generalizing m with
  | nil => rfl
  | cons w rest ih =>
    change load (byteExecute p r index rest (byteStep p r index w m))=_
    rw [ih,step_load]
    rfl

theorem readback (p : Instructions) (c : Call) (index : Indices) (accepted : check c index=true)
    (jobs : List Work) (active : ∀ w∈jobs,c.request.Active w) (unique : jobs.Nodup)
    (m : Bytes) (w : Work) (member : w∈jobs) (odd : Bool) :
    load (byteExecute p c.request index jobs m) (.other c.request.device) (destination c.request index w odd)=
      Native.expected p c.request w (load m) odd := by
  rw [execute_load]
  exact word_readback p c index accepted jobs active unique (load m) w member odd

theorem frame (p : Instructions) (r : Request) (index : Indices) (jobs : List Work) (m : Bytes)
    (region : Region) (address : Nat) (outside : ∀ w∈jobs,¬Writes r index w region (address/4)) :
    byteExecute p r index jobs m region address=m region address := by
  have h := word_frame p r index jobs (load m) region (address/4) outside
  rw [←execute_load] at h
  have bytes := congrArg (fun word => encode word ⟨address%4,Nat.mod_lt _ (by decide)⟩) h
  dsimp only [encode,load] at bytes
  rw [VeriTile.Bench.Repairs.RotaryEmb.Physical.original_bytes,
    VeriTile.Bench.Repairs.RotaryEmb.Physical.original_bytes] at bytes
  simpa [show 4*(address/4)+address%4=address by omega] using bytes

theorem output_bytes (p : Instructions) (c : Call) (index : Indices) (accepted : check c index=true)
    (jobs : List Work) (active : ∀ w∈jobs,c.request.Active w) (unique : jobs.Nodup)
    (m : Bytes) (w : Work) (member : w∈jobs) (odd : Bool) (i : Fin 4) :
    byteExecute p c.request index jobs m (.other c.request.device) (4*destination c.request index w odd+i.val)=
      encode (Native.expected p c.request w (load m) odd) i := by
  have h := congrArg (fun value => encode value i) (readback p c index accepted jobs active unique m w member odd)
  dsimp only [encode,load] at h
  rw [VeriTile.Bench.Repairs.RotaryEmb.Physical.original_bytes] at h
  exact h

#axiomsClean execute_load
#axiomsClean readback
#axiomsClean output_bytes
#axiomsClean frame
end VeriTile.Bench.Repairs.RotaryEmbNopad.Physical
