import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Bound

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

def outputWidth (fp8 : Bool) : Nat := if fp8 then 1 else 2
def Apart (a n b m : Nat) : Prop := a+n≤b ∨ b+m≤a

/-- Local accesses to one common physical store. All preservation laws are
for one one/two-byte output write; none mention a kernel, tile or result. -/
structure Access (α Store Byte : Type) where
  read32 : Store → Nat → α
  read16 : Store → Nat → α
  readE5 : Store → Nat → α
  read8 : Store → Nat → BitVec 8
  byte : Store → Nat → Byte
  write : Bool → Store → Nat → α → Store
  storable : Bool → α → Prop
  write16_same : ∀ s a v,storable Bool.false v → read16 (write Bool.false s a v) a=v
  writeE5_same : ∀ s a v,storable Bool.true v → readE5 (write Bool.true s a v) a=v
  read32_apart : ∀ f s a b v,Apart a (outputWidth f) b 4 → read32 (write f s a v) b=read32 s b
  read16_apart : ∀ f s a b v,Apart a (outputWidth f) b 2 → read16 (write f s a v) b=read16 s b
  readE5_apart : ∀ f s a b v,Apart a (outputWidth f) b 1 → readE5 (write f s a v) b=readE5 s b
  read8_apart : ∀ f s a b v,Apart a (outputWidth f) b 1 → read8 (write f s a v) b=read8 s b
  byte_apart : ∀ f s a b v,(b<a ∨ a+outputWidth f≤b) → byte (write f s a v) b=byte s b

def Access.read (access : Access α Store Byte) (fp8 : Bool) (s : Store) (a : Nat) : α :=
  if fp8 then access.readE5 s a else access.read16 s a

theorem read_write_same (access : Access α Store Byte) (fp8 : Bool) (s : Store) (a : Nat) (v : α)
    (valid : access.storable fp8 v) :
    access.read fp8 (access.write fp8 s a v) a=v := by
  cases fp8
  · exact access.write16_same s a v valid
  · exact access.writeE5_same s a v valid

theorem read_write_apart (access : Access α Store Byte) (out input : Bool)
    (s : Store) (a b : Nat) (v : α) (apart : Apart a (outputWidth out) b (outputWidth input)) :
    access.read input (access.write out s a v) b=access.read input s b := by
  cases input
  · exact access.read16_apart out s a b v apart
  · exact access.readE5_apart out s a b v apart

structure Write (α : Type) where
  fp8 : Bool
  address : Nat
  value : α

def run (access : Access α Store Byte) (events : List (Write α)) (s : Store) : Store :=
  events.foldl (fun t e => access.write e.fp8 t e.address e.value) s

theorem run_append (access : Access α Store Byte) (xs ys : List (Write α)) (s : Store) :
    run access (xs++ys) s=run access ys (run access xs s) := List.foldl_append

theorem run_byte_frame (access : Access α Store Byte) (events : List (Write α))
    (s : Store) (address : Nat)
    (outside : ∀ e∈events,address<e.address ∨ e.address+outputWidth e.fp8≤address) :
    access.byte (run access events s) address=access.byte s address := by
  induction events generalizing s with
  | nil => rfl
  | cons e rest ih =>
    rw [run,List.foldl_cons,←run,ih]
    · exact access.byte_apart _ _ _ _ _ (outside e (by simp))
    · exact fun e he => outside e (List.mem_cons_of_mem _ he)

theorem run_read_frame (access : Access α Store Byte) (events : List (Write α))
    (fp8 : Bool) (s : Store) (address : Nat)
    (outside : ∀ e∈events,Apart e.address (outputWidth e.fp8) address (outputWidth fp8)) :
    access.read fp8 (run access events s) address=access.read fp8 s address := by
  induction events generalizing s with
  | nil => rfl
  | cons e rest ih =>
    rw [run,List.foldl_cons,←run,ih]
    · exact read_write_apart access e.fp8 fp8 s e.address address e.value (outside e (by simp))
    · exact fun e he => outside e (List.mem_cons_of_mem _ he)

theorem run_readback (access : Access α Store Byte) (events : List (Write α))
    (s : Store) (before after : List (Write α)) (e : Write α)
    (partition : events=before++e::after)
    (valid : access.storable e.fp8 e.value)
    (apart : ∀ other∈after,Apart other.address (outputWidth other.fp8) e.address (outputWidth e.fp8)) :
    access.read e.fp8 (run access events s) e.address=e.value := by
  rw [partition,run_append,run,List.foldl_cons,←run,run_read_frame access _ _ _ _ apart]
  exact read_write_same access _ _ _ _ valid

#axiomsClean read_write_same
#axiomsClean read_write_apart
#axiomsClean run_byte_frame
#axiomsClean run_readback
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
