import bench.tritonbench_g.rms_matmul_rbe.Proofs.QKVReadback
import bench.tritonbench_g.rms_matmul_rbe.Proofs.MathArithmetic

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.RawGrid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

abbrev Memory (α : Type) := RegionName → Nat → Cell α

def state (template : State α) (mem : Memory α) (pid : Nat → Nat) : State α :=
  {template with mem:=mem,pids:=pid}

/-- The complete Q→K→V shared source runs at each scheduled program ID. Only
memory is carried between programs; scalar bindings come from the raw template. -/
def execute {α : Type} [Inhabited α] (a : Arithmetic α) (native fp8 : Bool)
    (p : Fin 3 → Arguments) (template : State α) : List (Nat → Nat) → Memory α → Option (Memory α)
  | [],mem => some mem
  | pid::rest,mem => do
    let final ← run a.model (qkvCode native fp8 p) (state template mem pid)
    execute a native fp8 p template rest final.mem

def step {α : Type} [Inhabited α] (a : Arithmetic α) (native fp8 : Bool)
    (p : Fin 3 → Arguments) (template : State α) (epsilon theta : α)
    (pid : Nat → Nat) (mem : Memory α) : Memory α :=
  (qkvWrite a native fp8 p (state template mem pid) epsilon theta).mem

/-- Independent ordered masked writes for all Q/K/V programs. -/
def reference {α : Type} [Inhabited α] (a : Arithmetic α) (native fp8 : Bool)
    (p : Fin 3 → Arguments) (template : State α) (epsilon theta : α) :
    List (Nat → Nat) → Memory α → Memory α
  | [],mem => mem
  | pid::rest,mem => reference a native fp8 p template epsilon theta rest
      (step a native fp8 p template epsilon theta pid mem)

def Outside (p : Fin 3 → Arguments) (order : List (Nat → Nat)) (r : RegionName) (o : Nat) : Prop :=
  ∀ pid ∈ order, ∀ phase : Fin 3,
    r≠(p phase).OUT ∨ ∀ i,
      outputActive (qkvArgument p phase) (pid 1 / columns (qkvArgument p phase))
        (pid 1 % columns (qkvArgument p phase)) i →
      o≠outputOffset (qkvArgument p phase) (pid 0) (pid 1 / columns (qkvArgument p phase))
        (pid 1 % columns (qkvArgument p phase)) i

theorem qkv_int8 {α : Type} [Inhabited α] (a : Arithmetic α) (native fp8 : Bool)
    (p : Fin 3 → Arguments) (s : State α) (epsilon theta : α) (bytes : Int8Memory s) :
    Int8Memory (qkvWrite a native fp8 p s epsilon theta) :=
  calleeWrite_int8 a native (selectedRotaryPlan 2) fp8 fp8 (qkvArgument p 2) s _ epsilon theta
    (calleeWrite_int8 a native (selectedRotaryPlan 1) fp8 fp8 (qkvArgument p 1) s _ epsilon theta
      (calleeWrite_int8 a native (selectedRotaryPlan 0) fp8 Bool.false (qkvArgument p 0) s s epsilon theta bytes))

theorem complete {α : Type} [Inhabited α] (a : Arithmetic α) (zero : ZeroLaws a)
    (native fp8 : Bool) (p : Fin 3 → Arguments) (template : State α) (epsilon theta : α)
    (he : template.regs .real [] "EPS"=some (fun _ => epsilon))
    (ht : template.regs .real [] "THETA"=some (fun _ => theta))
    (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT)
    (order : List (Nat → Nat)) (mem : Memory α)
    (bytes : Int8Memory {template with mem:=mem}) :
    execute a native fp8 p template order mem=
      some (reference a native fp8 p template epsilon theta order mem) := by
  induction order generalizing mem with
  | nil => rfl
  | cons pid rest ih =>
    obtain ⟨final,success,memory,_,_,_⟩ := qkv_complete a zero native fp8 p
      (state template mem pid) epsilon theta he ht bytes even apart
    simp only [execute,success]
    change execute a native fp8 p template rest final.mem = _
    rw [memory]
    exact ih _ (qkv_int8 a native fp8 p (state template mem pid) epsilon theta bytes)

theorem step_frame {α : Type} [Inhabited α] (a : Arithmetic α) (native fp8 : Bool)
    (p : Fin 3 → Arguments) (template : State α) (epsilon theta : α)
    (pid : Nat → Nat) (mem : Memory α) (r : RegionName) (o : Nat)
    (outside : Outside p [pid] r o) :
    step a native fp8 p template epsilon theta pid mem r o=mem r o := by
  have miss (phase : Fin 3) := outside pid (by simp) phase
  simp only [step,qkvWrite,calleeWrite,programRow,programCol,state]
  rw [tileWrite_frame a fp8 (qkvArgument p 2) _ _ _ _ _ r o (miss 2),
    tileWrite_frame a fp8 (qkvArgument p 1) _ _ _ _ _ r o (miss 1),
    tileWrite_frame a Bool.false (qkvArgument p 0) _ _ _ _ _ r o (miss 0)]

theorem reference_frame {α : Type} [Inhabited α] (a : Arithmetic α) (native fp8 : Bool)
    (p : Fin 3 → Arguments) (template : State α) (epsilon theta : α)
    (order : List (Nat → Nat)) (mem : Memory α) (r : RegionName) (o : Nat)
    (outside : Outside p order r o) :
    reference a native fp8 p template epsilon theta order mem r o=mem r o := by
  induction order generalizing mem with
  | nil => rfl
  | cons pid rest ih =>
    rw [reference,ih _ (fun job hj phase => outside job (List.mem_cons_of_mem _ hj) phase)]
    apply step_frame
    intro job hj phase
    exact outside job (List.mem_cons.mpr (Or.inl (List.mem_singleton.mp hj))) phase

theorem callee_mathematical (plan : RotaryPlan) (fp8 : Bool) (p : Arguments)
    (s : State ℝ) (epsilon theta : HostScalar) :
    calleeTarget mathArithmetic Bool.false plan fp8 p s epsilon.value theta.value=
      mathematicalTile fp8 p s epsilon theta := by
  simpa only [calleeTarget,boundTarget,if_false,math_scalar] using
    bound_mathematical plan fp8 p epsilon theta s

/-- Explicit independent Real targets at every scheduled Q/K/V masked write. -/
def mathematicalStep (fp8 : Bool) (p : Fin 3 → Arguments) (template : State ℝ)
    (epsilon theta : HostScalar) (pid : Nat → Nat) (mem : Memory ℝ) : Memory ℝ :=
  let s := state template mem pid
  let put := fun (phase : Fin 3) (current : State ℝ) =>
    let arg := qkvArgument p phase
    tileWrite mathArithmetic (qkvOutputFP8 fp8 phase) arg (pid 0)
      (programRow arg s) (programCol arg s) (mathematicalTile fp8 arg s epsilon theta) current
  (put 2 (put 1 (put 0 s))).mem

def mathematical (fp8 : Bool) (p : Fin 3 → Arguments) (template : State ℝ)
    (epsilon theta : HostScalar) : List (Nat → Nat) → Memory ℝ → Memory ℝ
  | [],mem => mem
  | pid::rest,mem => mathematical fp8 p template epsilon theta rest
      (mathematicalStep fp8 p template epsilon theta pid mem)

theorem step_mathematical (fp8 : Bool) (p : Fin 3 → Arguments) (template : State ℝ)
    (epsilon theta : HostScalar) (pid : Nat → Nat) (mem : Memory ℝ) :
    step mathArithmetic Bool.false fp8 p template epsilon.value theta.value pid mem=
      mathematicalStep fp8 p template epsilon theta pid mem := by
  simp only [step,qkvWrite,calleeWrite,callee_mathematical,mathematicalStep,
    qkvOutputFP8,show (0:Fin 3).val=0 from rfl,show (1:Fin 3).val=1 from rfl,
    show (2:Fin 3).val=2 from rfl,if_neg (by decide : ¬1=0),if_neg (by decide : ¬2=0)]
  rfl

theorem reference_mathematical (fp8 : Bool) (p : Fin 3 → Arguments) (template : State ℝ)
    (epsilon theta : HostScalar) (order : List (Nat → Nat)) (mem : Memory ℝ) :
    reference mathArithmetic Bool.false fp8 p template epsilon.value theta.value order mem=
      mathematical fp8 p template epsilon theta order mem := by
  induction order generalizing mem with
  | nil => rfl
  | cons pid rest ih => simp only [reference,mathematical,step_mathematical,ih]

#axiomsClean complete
#axiomsClean reference_frame
#axiomsClean reference_mathematical
#specNonCircular reference avoiding [execute,run,qkvCode,sourceTileValues,nativeTileValues]
#specNonCircular mathematical avoiding [execute,run,qkvCode,sourceTileValues,nativeTileValues]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.RawGrid
