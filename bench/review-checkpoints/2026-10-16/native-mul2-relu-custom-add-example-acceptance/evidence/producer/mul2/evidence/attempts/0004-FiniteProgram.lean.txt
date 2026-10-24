import bench.tritonbench_g.triton_mul2.Proofs.FiniteMemory

namespace TritonMul2Finite
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
noncomputable section
open Classical

/-- These are interpreter observations, not claims about a native exception,
race winner, rollback, or memory after a failed/racing launch. -/
inductive Issue where
  | inputAccess (pid lane address : Nat)
  | missingInputBytes (pid lane address : Nat)
  | outputAccess (pid lane address : Nat)
  | outputOwnershipConflict (pid lane address : Nat)
  | sourceElaboration
  deriving DecidableEq, Repr
def requireSome (value : Option α) (issue : Issue) : Except Issue α :=
  match value with | none => .error issue | some value => .ok value
@[simp] theorem bind_ok (value : α) (f : α → Except ε β) :
    (Except.ok value >>= f)=f value := rfl
@[simp] theorem bind_error (issue : ε) (f : α → Except ε β) :
    (Except.error issue >>= f)=Except.error issue := rfl
@[simp] theorem pure_ok (value : α) : (pure value : Except ε α)=.ok value := rfl

structure Loaded (d : Storage) where
  lane : Nat
  inputAddress : Nat
  outputAddress : Nat
  deviceAddress : Nat
  value : Word d

/-- Complete active tile loads occur before any stores. Strides are absent
from source addressing: actual extracted ABI pointers plus signed finite offsets.
Reads resolve device aliases into the same canonical backing-byte memory. -/
def loadTile (d : Storage) (p : Layout) (snapshot : ByteMemory) (pid : Nat)
    (inputOffset outputOffset : Nat → BitVec 32) (mask : Nat → Bool) :
    List Nat → Except Issue (List (Loaded d))
  | [] => .ok []
  | lane::rest =>
    if mask lane then
      let address := (p.inputPointer + BitVec.ofInt 64 (inputOffset lane).toInt * BitVec.ofNat 64 d.bytes).toNat
      if ¬(InRegion p.deviceInput address d.bytes ∧ p.access.Readable address d.bytes) then
        .error (.inputAccess pid lane address)
      else do
        let value ← requireSome (readWord d (deviceMemory p.access snapshot) address) (.missingInputBytes pid lane address)
        let tail ← loadTile d p snapshot pid inputOffset outputOffset mask rest
        let outputAddress := (p.outputPointer + BitVec.ofInt 64 (outputOffset lane).toInt * BitVec.ofNat 64 d.bytes).toNat
        pure (⟨lane,address,p.outputBacking outputAddress,outputAddress,value⟩::tail)
    else loadTile d p snapshot pid inputOffset outputOffset mask rest

def tileWrites (d : Storage) (compute : Word d → Word d) (p : Layout) (pid : Nat) :
    List (Loaded d) → List Nat → Except Issue (List ByteWrite × List Nat)
  | [],seen => .ok ([],seen)
  | loaded::rest,seen =>
    if ¬(InRegion p.deviceOutput loaded.deviceAddress d.bytes ∧
        loaded.outputAddress=p.outputBacking loaded.deviceAddress ∧
        p.access.Resolves loaded.deviceAddress loaded.outputAddress d.bytes ∧
        p.access.Writable loaded.deviceAddress d.bytes) then
      .error (.outputAccess pid loaded.lane loaded.deviceAddress)
    else if ∃ k, k<d.bytes ∧ loaded.outputAddress+k∈seen then
      .error (.outputOwnershipConflict pid loaded.lane loaded.deviceAddress)
    else do
      let head := wordWrites d pid loaded.lane loaded.inputAddress loaded.outputAddress loaded.deviceAddress (compute loaded.value)
      let (tail,owners) ← tileWrites d compute p pid rest (head.map ByteWrite.outputAddress++seen)
      pure (head++tail,owners)

def programPlan (b : Numerics) (d : Storage) (p : Layout) (n block pid : Nat)
    (snapshot : ByteMemory) (seen : List Nat) : Except Issue (List ByteWrite × List Nat) := do
  let loaded ← loadTile d p snapshot pid (offset32 block pid) (offset32 block pid)
    (active32 n block pid) (List.range block)
  tileWrites d (result b d) p pid loaded seen

structure State where
  memory : ByteMemory
  owners : List Nat

def executeProgram (b : Numerics) (d : Storage) (p : Layout) (n block pid : Nat)
    (state : State) : Except Issue State := do
  let loaded ← loadTile d p state.memory pid (offset32 block pid) (offset32 block pid)
    (active32 n block pid) (List.range block)
  let (writes,owners) ← tileWrites d (fun x => stored d (source.eval b d x)) p pid loaded state.owners
  pure ⟨applyWrites writes state.memory,owners⟩

theorem program_correspondence (b : Numerics) (d : Storage) (p : Layout) (n block pid : Nat)
    (state : State) : executeProgram b d p n block pid state=
    (programPlan b d p n block pid state.memory state.owners).map
      (fun plan => ⟨applyWrites plan.1 state.memory,plan.2⟩) := by
  simp only [executeProgram,programPlan,source_scalar]
  cases loadTile d p state.memory pid (offset32 block pid) (offset32 block pid)
    (active32 n block pid) (List.range block) with
  | error issue => rfl
  | ok loaded => cases tileWrites d (result b d) p pid loaded state.owners <;> rfl

theorem tile_write_bounds (d : Storage) (compute : Word d → Word d) (p : Layout) (pid : Nat)
    (loaded : List (Loaded d)) (seen : List Nat) (writes : List ByteWrite) (owners : List Nat)
    (h : tileWrites d compute p pid loaded seen=.ok (writes,owners)) :
    ∀ w∈writes, p.output.Contains w.outputAddress := by
  induction loaded generalizing seen writes owners with
  | nil => cases h; simp
  | cons item rest ih =>
    simp only [tileWrites] at h
    split at h
    · cases h
    · rename_i valid
      split at h
      · cases h
      · cases ht : tileWrites d compute p pid rest
          ((wordWrites d pid item.lane item.inputAddress item.outputAddress item.deviceAddress (compute item.value)).map ByteWrite.outputAddress++seen) with
        | error issue => simp [ht] at h
        | ok tail =>
          rcases tail with ⟨tail,finalOwners⟩
          simp [ht] at h
          rcases h with ⟨rfl,rfl⟩
          intro w hw
          rcases List.mem_append.mp hw with hw|hw
          · exact word_write_bounds d pid item.lane item.inputAddress item.outputAddress item.deviceAddress
              (compute item.value) p.output (by
                rw [(not_not.mp valid).2.1]
                exact translated_backing_bounds p.output p.outputDeviceBase item.deviceAddress d.bytes (not_not.mp valid).1) w hw
          · exact ih _ _ _ ht w hw

theorem program_write_bounds (b : Numerics) (d : Storage) (p : Layout) (n block pid : Nat)
    (snapshot : ByteMemory) (seen : List Nat) (writes : List ByteWrite) (owners : List Nat)
    (h : programPlan b d p n block pid snapshot seen=.ok (writes,owners)) :
    ∀ w∈writes, p.output.Contains w.outputAddress := by
  unfold programPlan at h
  cases hl : loadTile d p snapshot pid (offset32 block pid) (offset32 block pid)
      (active32 n block pid) (List.range block) with
  | error issue => simp [hl] at h
  | ok loaded =>
    simp only [hl,bind_ok] at h
    exact tile_write_bounds d _ p pid loaded seen writes owners h

#specNonCircular programPlan avoiding [executeProgram,applyWrites,source,Expr.eval,stored,multiply]
#axiomsClean program_correspondence
#axiomsClean program_write_bounds
#guard_msgs (drop info) in
#auditModuleAxioms
end
end TritonMul2Finite
