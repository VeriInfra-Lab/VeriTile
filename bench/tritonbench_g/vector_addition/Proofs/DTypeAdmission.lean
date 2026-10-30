import bench.tritonbench_g.vector_addition.Proofs.DTypeScope

/-! A finite model of selected dtype gates in the frozen CUDA sm90 frontend.
This is not a semantics or correctness proof of Python, JIT compilation,
allocation, native lowering, device errors or GPU execution. Source audits and
observed exceptions remain separate evidence. Signature admission precedes
tensor-add promotion, including when both gates would reject in isolation. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.DTypeAdmission
open DTypeScope
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000

inductive Argument where
  | x | y | output | nElements
  deriving DecidableEq, Repr

def backendName : DType → String
  | .b1 => "int1" | .i8 => "int8" | .u8 => "uint8"
  | .i16 => "int16" | .u16 => "uint16"
  | .i32 => "int32" | .u32 => "uint32"
  | .i64 => "int64" | .u64 => "uint64"
  | .fp16 => "fp16" | .bf16 => "bf16" | .fp32 => "fp32" | .fp64 => "fp64"
  | .fp8e5 => "fp8e5" | .fp8e4nv => "fp8e4nv"
  | .fp8e5b16 => "fp8e5b16" | .fp8e4b8 => "fp8e4b8"

def supportedFP8 : List String := ["fp8e4b15", "fp8e4nv", "fp8e5"]

def scalarAdmitted (d : DType) : Bool :=
  if d.fp8 then decide (backendName d ∈ supportedFP8) else Bool.true

def fnuz : DType → Bool
  | .fp8e5b16 | .fp8e4b8 => Bool.true
  | _ => Bool.false

def integerBool (d : DType) : Bool := decide (0 < d.width)

def signature (x y : DType) : List (Argument × DType) :=
  [(.x, x), (.y, y), (.output, x), (.nElements, .i32)]

def firstRejected : List (Argument × DType) → Option (Argument × DType)
  | [] => none
  | entry :: rest => if scalarAdmitted entry.2 then firstRejected rest else some entry

inductive Outcome where
  | signatureRejected (argument : Argument) (dtype : DType)
  | additionRejected
  | passesDTypeGates (computation store : DType)
  deriving DecidableEq, Repr

inductive Stage where
  | signature | addition | admitted
  deriving DecidableEq, Repr

def Outcome.stage : Outcome → Stage
  | .signatureRejected _ _ => .signature
  | .additionRejected => .addition
  | .passesDTypeGates _ _ => .admitted

def admission (x y : DType) : Outcome :=
  match firstRejected (signature x y) with
  | some (arg, dtype) => .signatureRejected arg dtype
  | none => match promote x y with
    | none => .additionRejected
    | some dtype => .passesDTypeGates dtype (storeElement x)

theorem scalar_admitted_iff (d : DType) : scalarAdmitted d = !fnuz d := by
  cases d <;> decide

theorem signature_first (x y : DType) :
    firstRejected (signature x y) =
      if fnuz x then some (.x, x) else if fnuz y then some (.y, y) else none := by
  cases x <;> cases y <;> decide

theorem promotion_rejected_iff (x y : DType) :
    promote x y = none ↔
      (integerBool x = Bool.true ∧ y.fp8 = Bool.true) ∨
      (x.fp8 = Bool.true ∧ integerBool y = Bool.true) := by
  cases x <;> cases y <;> decide

theorem fnuz_first_stage (x y : DType) :
    (fnuz x = Bool.true → admission x y = .signatureRejected .x x) ∧
    (fnuz x = Bool.false → fnuz y = Bool.true → admission x y = .signatureRejected .y y) := by
  cases x <;> cases y <;> decide

theorem addition_rejected_iff (x y : DType) :
    admission x y = .additionRejected ↔
      fnuz x = Bool.false ∧ fnuz y = Bool.false ∧ promote x y = none := by
  cases x <;> cases y <;> decide

theorem admission_exhaustive (x y : DType) :
    (∃ arg d, admission x y = .signatureRejected arg d) ∨
    admission x y = .additionRejected ∨
    (∃ d, admission x y = .passesDTypeGates d (storeElement x)) := by
  cases h : firstRejected (signature x y) with
  | some entry =>
    rcases entry with ⟨arg, d⟩
    exact Or.inl ⟨arg, d, by simp [admission, h]⟩
  | none =>
    cases hp : promote x y with
    | none => exact Or.inr (Or.inl (by simp [admission, h, hp]))
    | some d => exact Or.inr (Or.inr ⟨d, by simp [admission, h, hp]⟩)

def pairs : List (DType × DType) :=
  profiles.flatMap fun x => profiles.map fun y => (x, y)

def stageRows (stage : Stage) : List (DType × DType) :=
  pairs.filter fun pair => decide ((admission pair.1 pair.2).stage = stage)

def crossRows (stage : Stage) (promotionSucceeds : Bool) : List (DType × DType) :=
  (stageRows stage).filter fun pair => (promote pair.1 pair.2).isSome == promotionSucceeds

theorem profiles_complete (d : DType) : d ∈ profiles := by cases d <;> decide

theorem pair_inventory : pairs.length = 289 ∧ pairs.Nodup := by decide

theorem stage_partition_counts :
    (stageRows .admitted).length = 189 ∧
    (stageRows .signature).length = 64 ∧
    (stageRows .addition).length = 36 := by decide

theorem standalone_promotion_counts :
    (pairs.filter fun pair => (promote pair.1 pair.2).isSome).length = 217 ∧
    (pairs.filter fun pair => (promote pair.1 pair.2).isNone).length = 72 := by decide

theorem stage_promotion_cross_counts :
    (crossRows .admitted Bool.true).length = 189 ∧
    (crossRows .addition Bool.false).length = 36 ∧
    (crossRows .signature Bool.false).length = 36 ∧
    (crossRows .signature Bool.true).length = 28 := by decide

theorem stage_rows_disjoint (a b : Stage) (different : a ≠ b) :
    List.Disjoint (stageRows a) (stageRows b) := by
  intro pair ha hb
  have aa : (admission pair.1 pair.2).stage = a := of_decide_eq_true (List.mem_filter.mp ha).2
  have bb : (admission pair.1 pair.2).stage = b := of_decide_eq_true (List.mem_filter.mp hb).2
  exact different (aa.symm.trans bb)

theorem all_pairs_classified (pair : DType × DType) :
    pair ∈ stageRows .signature ∨ pair ∈ stageRows .addition ∨ pair ∈ stageRows .admitted := by
  rcases pair with ⟨x, y⟩
  cases x <;> cases y <;> decide

#axiomsClean scalar_admitted_iff
#axiomsClean signature_first
#axiomsClean promotion_rejected_iff
#axiomsClean fnuz_first_stage
#axiomsClean addition_rejected_iff
#axiomsClean admission_exhaustive
#axiomsClean profiles_complete
#axiomsClean pair_inventory
#axiomsClean stage_partition_counts
#axiomsClean standalone_promotion_counts
#axiomsClean stage_promotion_cross_counts
#axiomsClean stage_rows_disjoint
#axiomsClean all_pairs_classified
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VectorAddition.DTypeAdmission
