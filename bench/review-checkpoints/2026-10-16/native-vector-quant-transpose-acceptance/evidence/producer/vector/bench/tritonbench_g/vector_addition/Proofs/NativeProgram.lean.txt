import bench.tritonbench_g.vector_addition.Proofs.NativeMemory

namespace VeriTile.Bench.TritonBenchG.VectorAddition.Native
open VeriTile.Triton DTypeScope TypedHost Finite
noncomputable section
open Classical
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000

/-- These are mathematical observations of an actual request. They assert no
native exception, rollback, poison value, or winner of a conflicting store. -/
inductive Issue where
  | inputAccess (left : Bool) (pid lane address : Nat)
  | missingBytes (left : Bool) (pid lane address : Nat)
  | inputEncoding (left : Bool) (pid lane address : Nat)
  | undefinedConversion (pid lane : Nat)
  | outputAccess (pid lane address : Nat)
  | ownershipConflict (pid lane address : Nat)
  deriving DecidableEq, Repr

def requireSome (value : Option α) (issue : Issue) : Except Issue α :=
  match value with | none => .error issue | some value => .ok value
@[simp] theorem bind_ok (value : α) (f : α → Except ε β) :
    (Except.ok value >>= f)=f value := rfl
@[simp] theorem bind_error (issue : ε) (f : α → Except ε β) :
    (Except.error issue >>= f)=Except.error issue := rfl
@[simp] theorem pure_ok (value : α) : (pure value : Except ε α)=.ok value := rfl

structure LoadedX (A : Type) where
  lane : Nat
  address : Nat
  value : A
structure Loaded (A : Type) where
  lane : Nat
  xAddress : Nat
  yAddress : Nat
  outputAddress : Nat
  x : A
  y : A

/-- Source x load statement, across the complete active tile. -/
def loadX {M : Type} (r : Route) (ops : Memory M (Carrier r)) (p : Layout)
    (memory : M) (pid : Nat) : List Nat → Except Issue (List (LoadedX (Carrier r)))
  | [] => .ok []
  | lane::rest =>
      let address := pointer p.x.data (bytes r.signature.1) pid lane
      if ¬p.x.Readable p.device address (bytes r.signature.1) then .error (.inputAccess Bool.true pid lane address)
      else do
        let value ← requireSome (ops.load memory r.signature.1 address) (.missingBytes Bool.true pid lane address)
        if ¬scalarInput r Bool.true value then .error (.inputEncoding Bool.true pid lane address)
        else
          let tail ← loadX r ops p memory pid rest
          pure (⟨lane,address,value⟩::tail)

/-- Source y load statement executes after all x loads and before any store. -/
def loadY {M : Type} (r : Route) (ops : Memory M (Carrier r)) (p : Layout)
    (memory : M) (pid : Nat) : List (LoadedX (Carrier r)) → Except Issue (List (Loaded (Carrier r)))
  | [] => .ok []
  | item::rest =>
      let address := pointer p.y.data (bytes r.signature.2) pid item.lane
      if ¬p.y.Readable p.device address (bytes r.signature.2) then .error (.inputAccess Bool.false pid item.lane address)
      else do
        let value ← requireSome (ops.load memory r.signature.2 address) (.missingBytes Bool.false pid item.lane address)
        if ¬scalarInput r Bool.false value then .error (.inputEncoding Bool.false pid item.lane address)
        else
          let tail ← loadY r ops p memory pid rest
          pure (⟨item.lane,item.address,address,pointer p.output.data (bytes r.signature.1) pid item.lane,item.value,value⟩::tail)

def span (address width : Nat) : List Nat := (List.range width).map (address+·)
def Conflict (address width : Nat) (seen : List Nat) : Prop := ∃ k, k<width ∧ address+k∈seen

/-- Independent initial-read value and actual output-address request plan. -/
def planWrites (b : Numerics) (r : Route) (p : Layout) (pid : Nat) :
    List (Loaded (Carrier r)) → List Nat → Except Issue (List (Write (Carrier r)) × List Nat)
  | [],seen => .ok ([],seen)
  | item::rest,seen =>
      if ¬scalarDefined b r item.x item.y then .error (.undefinedConversion pid item.lane)
      else if ¬p.output.Writable p.device item.outputAddress (bytes r.signature.1) then
        .error (.outputAccess pid item.lane item.outputAddress)
      else if Conflict item.outputAddress (bytes r.signature.1) seen then
        .error (.ownershipConflict pid item.lane item.outputAddress)
      else do
        let (tail,owners) ← planWrites b r p pid rest (span item.outputAddress (bytes r.signature.1)++seen)
        pure (⟨pid,item.lane,item.xAddress,item.yAddress,item.outputAddress,scalarResult b r item.x item.y⟩::tail,owners)

/-- Source values use the already verified typed scalar execution functions. -/
def sourceWrites (b : Numerics) (r : Route) (p : Layout) (pid : Nat) :
    List (Loaded (Carrier r)) → List Nat → Except Issue (List (Write (Carrier r)) × List Nat)
  | [],seen => .ok ([],seen)
  | item::rest,seen =>
      if ¬scalarDefined b r item.x item.y then .error (.undefinedConversion pid item.lane)
      else if ¬p.output.Writable p.device item.outputAddress (bytes r.signature.1) then
        .error (.outputAccess pid item.lane item.outputAddress)
      else if Conflict item.outputAddress (bytes r.signature.1) seen then
        .error (.ownershipConflict pid item.lane item.outputAddress)
      else do
        let (tail,owners) ← sourceWrites b r p pid rest (span item.outputAddress (bytes r.signature.1)++seen)
        pure (⟨pid,item.lane,item.xAddress,item.yAddress,item.outputAddress,Finite.scalarSource b r item.x item.y⟩::tail,owners)

theorem source_writes_correspondence (b : Numerics) (r : Route) (p : Layout) (pid : Nat)
    (loaded : List (Loaded (Carrier r))) (seen : List Nat) :
    sourceWrites b r p pid loaded seen=planWrites b r p pid loaded seen := by
  induction loaded generalizing seen with
  | nil => rfl
  | cons item rest ih =>
      simp only [sourceWrites,planWrites]
      split
      · rfl
      · rename_i hd
        rw [scalar_defined_correspondence b r _ _ (not_not.mp hd),ih]

def programPlan {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (n : Nat) (snapshot : M) (pid : Nat) (seen : List Nat) :
    Except Issue (List (Write (Carrier r)) × List Nat) := do
  let xs ← loadX r ops p snapshot pid (activeTile n pid)
  let loaded ← loadY r ops p snapshot pid xs
  planWrites b r p pid loaded seen

structure State (M : Type) where
  memory : M
  owners : List Nat

/-- Full tile loads, typed arithmetic, and ordinary actual-address stores. -/
def executeProgram {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (n pid : Nat) (state : State M) : Except Issue (State M) := do
  let xs ← loadX r ops p state.memory pid (activeTile n pid)
  let loaded ← loadY r ops p state.memory pid xs
  let (writes,owners) ← sourceWrites b r p pid loaded state.owners
  pure ⟨applyWrites ops r.signature.1 writes state.memory,owners⟩

theorem program_correspondence {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (n pid : Nat) (state : State M) :
    executeProgram r ops b p n pid state=(programPlan r ops b p n state.memory pid state.owners).map
      (fun plan => ⟨applyWrites ops r.signature.1 plan.1 state.memory,plan.2⟩) := by
  simp only [executeProgram,programPlan,source_writes_correspondence]
  cases loadX r ops p state.memory pid (activeTile n pid) with
  | error issue => rfl
  | ok xs =>
      simp only [bind_ok]
      cases loadY r ops p state.memory pid xs with
      | error issue => rfl
      | ok loaded =>
          simp only [bind_ok]
          cases planWrites b r p pid loaded state.owners <;> rfl

theorem load_x_frame {M : Type} (r : Route) (ops : Memory M (Carrier r)) (p : Layout)
    (left right : M) (same : OutsideOutput ops p left right) (fresh : p.x.Disjoint p.output)
    (pid : Nat) (lanes : List Nat) : loadX r ops p left pid lanes=loadX r ops p right pid lanes := by
  induction lanes with
  | nil => rfl
  | cons lane rest ih =>
      simp only [loadX]
      split
      · rfl
      · rename_i valid
        rw [load_outside ops p left right p.x _ _ same fresh (not_not.mp valid).1,ih]

theorem load_y_frame {M : Type} (r : Route) (ops : Memory M (Carrier r)) (p : Layout)
    (left right : M) (same : OutsideOutput ops p left right) (fresh : p.y.Disjoint p.output)
    (pid : Nat) (xs : List (LoadedX (Carrier r))) : loadY r ops p left pid xs=loadY r ops p right pid xs := by
  induction xs with
  | nil => rfl
  | cons item rest ih =>
      simp only [loadY]
      split
      · rfl
      · rename_i valid
        rw [load_outside ops p left right p.y _ _ same fresh (not_not.mp valid).1,ih]

theorem program_plan_frame {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (n : Nat) (left right : M) (same : OutsideOutput ops p left right)
    (fresh : p.Fresh) (pid : Nat) (seen : List Nat) :
    programPlan r ops b p n left pid seen=programPlan r ops b p n right pid seen := by
  rw [programPlan,programPlan,load_x_frame r ops p left right same fresh.1]
  cases loadX r ops p right pid (activeTile n pid) with
  | error issue => rfl
  | ok xs => simp only [bind_ok,load_y_frame r ops p left right same fresh.2]

theorem plan_write_bounds (b : Numerics) (r : Route) (p : Layout) (pid : Nat)
    (loaded : List (Loaded (Carrier r))) (seen : List Nat)
    (writes : List (Write (Carrier r))) (owners : List Nat)
    (h : planWrites b r p pid loaded seen=.ok (writes,owners)) :
    ∀ w∈writes, p.output.Inside w.address (bytes r.signature.1) := by
  induction loaded generalizing seen writes owners with
  | nil => cases h; simp
  | cons item rest ih =>
      simp only [planWrites] at h
      split at h
      · cases h
      · split at h
        · cases h
        · rename_i valid
          split at h
          · cases h
          · cases ht : planWrites b r p pid rest (span item.outputAddress (bytes r.signature.1)++seen) with
            | error issue => simp [ht] at h
            | ok tail =>
                rcases tail with ⟨tail,finalOwners⟩
                simp [ht] at h
                rcases h with ⟨rfl,rfl⟩
                intro w hw
                rcases List.mem_cons.1 hw with rfl|hw
                · exact (not_not.mp valid).1
                · exact ih _ _ _ ht w hw

theorem program_write_bounds {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (n : Nat) (snapshot : M) (pid : Nat) (seen : List Nat)
    (writes : List (Write (Carrier r))) (owners : List Nat)
    (h : programPlan r ops b p n snapshot pid seen=.ok (writes,owners)) :
    ∀ w∈writes, p.output.Inside w.address (bytes r.signature.1) := by
  unfold programPlan at h
  cases hx : loadX r ops p snapshot pid (activeTile n pid) with
  | error issue => simp [hx] at h
  | ok xs =>
      cases hy : loadY r ops p snapshot pid xs with
      | error issue => simp [hx,hy] at h
      | ok loaded =>
          simp only [hx,hy,bind_ok] at h
          exact plan_write_bounds b r p pid loaded seen writes owners h

theorem no_access_program {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (snapshot : M) (pid : Nat) (seen : List Nat) :
    programPlan r ops b p (2^31) snapshot pid seen=.ok ([],seen) := by
  rw [programPlan,active_tile_empty_at_two31]
  rfl

#specNonCircular programPlan avoiding [executeProgram,sourceWrites,applyWrites]
#axiomsClean source_writes_correspondence
#axiomsClean program_correspondence
#axiomsClean program_plan_frame
#axiomsClean program_write_bounds
#axiomsClean no_access_program
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.Native
