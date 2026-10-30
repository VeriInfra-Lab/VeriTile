import bench.tritonbench_g.add_value.Proofs.NativeMemory

namespace AddValueNative
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.VectorAddition
open DTypeScope TypedHost Finite
noncomputable section
open Classical
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000

/-- Mathematical observations; no native exception, rollback, poison value,
postfailure memory, or race winner is asserted. -/
inductive Issue where
  | inputAccess (pid lane address : Nat)
  | missingBytes (pid lane address : Nat)
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

structure Loaded (A : Type) where
  lane : Nat
  xAddress : Nat
  outputAddress : Nat
  value : A

/-- Exactly one source input pointer. Every active tile word is loaded before
any scalar operation or output store. The carrier is already the exact finite
storage word, including all raw boolean bytes. -/
def loadX {M : Type} (d : Storage) (ops : Memory M (Word (.storage d))) (p : Layout)
    (memory : M) (pid : Nat) : List Nat → Except Issue (List (Loaded (Word (.storage d))))
  | [] => .ok []
  | lane::rest =>
      let address := pointer p.x.data (bytes (dtype d)) pid lane
      if ¬p.x.Readable p.device address (bytes (dtype d)) then .error (.inputAccess pid lane address)
      else do
        let value ← requireSome (ops.load memory (dtype d) address) (.missingBytes pid lane address)
        let tail ← loadX d ops p memory pid rest
        pure (⟨lane,address,pointer p.output.data (bytes (dtype d)) pid lane,value⟩::tail)

def span (address width : Nat) : List Nat := (List.range width).map (address+·)
def Conflict (address width : Nat) (seen : List Nat) : Prop := ∃ k, k<width ∧ address+k∈seen

/-- Internal typed instruction fold. The public program fixes eval to the
actual source expression; the independent plan fixes it to the proved formula.
No caller supplies a final output or kernel success. -/
def buildWrites (d : Storage) (eval : Word (.storage d) → Option (Word (.storage d)))
    (p : Layout) (pid : Nat) : List (Loaded (Word (.storage d))) → List Nat →
    Except Issue (List (Write (Word (.storage d))) × List Nat)
  | [],seen => .ok ([],seen)
  | item::rest,seen => match eval item.value with
    | none => .error (.undefinedConversion pid item.lane)
    | some value =>
      if ¬p.output.Writable p.device item.outputAddress (bytes (dtype d)) then
        .error (.outputAccess pid item.lane item.outputAddress)
      else if Conflict item.outputAddress (bytes (dtype d)) seen then
        .error (.ownershipConflict pid item.lane item.outputAddress)
      else do
        let (tail,owners) ← buildWrites d eval p pid rest (span item.outputAddress (bytes (dtype d))++seen)
        pure (⟨pid,item.lane,item.xAddress,item.outputAddress,value⟩::tail,owners)

def planWrites (b : Numerics) (d : Storage) := buildWrites d (fun value => result b d value argument)
def sourceWrites (b : Numerics) (d : Storage) := buildWrites d (fun value => (source d).eval b value argument)

theorem source_writes_correspondence (b : Numerics) (d : Storage) :
    sourceWrites b d=planWrites b d := by
  unfold sourceWrites planWrites
  congr 1
  funext value
  exact source_scalar b d value argument

def programPlan {M : Type} (d : Storage) (ops : Memory M (Word (.storage d))) (b : Numerics)
    (p : Layout) (n : Nat) (snapshot : M) (pid : Nat) (seen : List Nat) :
    Except Issue (List (Write (Word (.storage d))) × List Nat) := do
  let loaded ← loadX d ops p snapshot pid (activeTile n pid)
  planWrites b d p pid loaded seen

structure State (M : Type) where
  memory : M
  owners : List Nat

def executeProgram {M : Type} (d : Storage) (ops : Memory M (Word (.storage d))) (b : Numerics)
    (p : Layout) (n pid : Nat) (state : State M) : Except Issue (State M) := do
  let loaded ← loadX d ops p state.memory pid (activeTile n pid)
  let (writes,owners) ← sourceWrites b d p pid loaded state.owners
  pure ⟨applyWrites ops (dtype d) writes state.memory,owners⟩

theorem program_correspondence {M : Type} (d : Storage) (ops : Memory M (Word (.storage d))) (b : Numerics)
    (p : Layout) (n pid : Nat) (state : State M) :
    executeProgram d ops b p n pid state=(programPlan d ops b p n state.memory pid state.owners).map
      (fun plan => ⟨applyWrites ops (dtype d) plan.1 state.memory,plan.2⟩) := by
  simp only [executeProgram,programPlan,source_writes_correspondence]
  cases loadX d ops p state.memory pid (activeTile n pid) with
  | error issue => rfl
  | ok loaded =>
      simp only [bind_ok]
      cases planWrites b d p pid loaded state.owners <;> rfl

theorem load_x_frame {M : Type} (d : Storage) (ops : Memory M (Word (.storage d))) (p : Layout)
    (left right : M) (same : OutsideOutput ops p left right) (fresh : p.x.Disjoint p.output)
    (pid : Nat) (lanes : List Nat) : loadX d ops p left pid lanes=loadX d ops p right pid lanes := by
  induction lanes with
  | nil => rfl
  | cons lane rest ih =>
      simp only [loadX]
      split
      · rfl
      · rename_i valid
        rw [load_outside ops p left right p.x _ _ same fresh (not_not.mp valid).1,ih]

theorem program_plan_frame {M : Type} (d : Storage) (ops : Memory M (Word (.storage d))) (b : Numerics)
    (p : Layout) (n : Nat) (left right : M) (same : OutsideOutput ops p left right)
    (fresh : p.Fresh) (pid : Nat) (seen : List Nat) :
    programPlan d ops b p n left pid seen=programPlan d ops b p n right pid seen := by
  rw [programPlan,programPlan,load_x_frame d ops p left right same fresh]

theorem build_write_bounds (d : Storage) (eval : Word (.storage d) → Option (Word (.storage d)))
    (p : Layout) (pid : Nat) (loaded : List (Loaded (Word (.storage d)))) (seen : List Nat)
    (writes : List (Write (Word (.storage d)))) (owners : List Nat)
    (h : buildWrites d eval p pid loaded seen=.ok (writes,owners)) :
    ∀ w∈writes, p.output.Inside w.address (bytes (dtype d)) := by
  induction loaded generalizing seen writes owners with
  | nil => cases h; simp
  | cons item rest ih =>
      simp only [buildWrites] at h
      cases hv : eval item.value with
      | none => simp [hv] at h
      | some value =>
        simp only [hv] at h
        split at h
        · cases h
        · rename_i valid
          split at h
          · cases h
          · cases ht : buildWrites d eval p pid rest (span item.outputAddress (bytes (dtype d))++seen) with
            | error issue => simp [ht] at h
            | ok tail =>
                rcases tail with ⟨tail,finalOwners⟩
                simp [ht] at h
                rcases h with ⟨rfl,rfl⟩
                intro w hw
                rcases List.mem_cons.1 hw with rfl|hw
                · exact (not_not.mp valid).1
                · exact ih _ _ _ ht w hw

theorem program_write_bounds {M : Type} (d : Storage) (ops : Memory M (Word (.storage d))) (b : Numerics)
    (p : Layout) (n : Nat) (snapshot : M) (pid : Nat) (seen : List Nat)
    (writes : List (Write (Word (.storage d)))) (owners : List Nat)
    (h : programPlan d ops b p n snapshot pid seen=.ok (writes,owners)) :
    ∀ w∈writes, p.output.Inside w.address (bytes (dtype d)) := by
  unfold programPlan at h
  cases hx : loadX d ops p snapshot pid (activeTile n pid) with
  | error issue => simp [hx] at h
  | ok loaded =>
      simp only [hx,bind_ok] at h
      exact build_write_bounds d _ p pid loaded seen writes owners h

theorem no_access_program {M : Type} (d : Storage) (ops : Memory M (Word (.storage d))) (b : Numerics)
    (p : Layout) (snapshot : M) (pid : Nat) (seen : List Nat) :
    programPlan d ops b p (2^31) snapshot pid seen=.ok ([],seen) := by
  rw [programPlan,active_tile_empty_at_two31]
  rfl

#specNonCircular programPlan avoiding [executeProgram,sourceWrites,applyWrites,source,Expr.eval]
#axiomsClean source_writes_correspondence
#axiomsClean program_correspondence
#axiomsClean program_plan_frame
#axiomsClean program_write_bounds
#axiomsClean no_access_program
#guard_msgs (drop info) in
#auditModuleAxioms
end
end AddValueNative
