import bench.tritonbench_g.relu_triton_kernel.Proofs.FP64Grid

/-! Finite host dispatch on the frozen seventeen-profile frontend domain.
The old twelve logical routes are invoked unchanged. The wide route invokes
the new statement interpreter. Rejections stop at the launch primitive and
cannot produce a return event. This is under the named frontend/allocation
primitive boundary, not a theorem about arbitrary Python/compiler internals. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.FP64Host
open VeriTile.Triton GatedGrid HostWrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

inductive Admission where
  | typed (profile : TypedValue.Profile)
  | fp64
  | comparisonRejected
  | signatureRejected
  | outsideDomain
  deriving DecidableEq, Repr

def admission (dtype : String) : Admission :=
  match selectProfile dtype with
  | some p => .typed p
  | none => match dtype with
    | "float64" => .fp64
    | "float8_e4m3fn" | "float8_e5m2" => .comparisonRejected
    | "float8_e4m3fnuz" | "float8_e5m2fnuz" => .signatureRejected
    | _ => .outsideDomain

def declaredDTypes : List String :=
  ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
   "bool", "float16", "bfloat16", "float32", "float64",
   "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"]

def admitted : Admission → Bool
  | .typed _ | .fp64 => Bool.true
  | _ => Bool.false

theorem finite_inventory :
    declaredDTypes.length = 17 ∧
    (declaredDTypes.filter (fun d => admitted (admission d))).length = 13 ∧
    (declaredDTypes.filter (fun d => admission d == .comparisonRejected)).length = 2 ∧
    (declaredDTypes.filter (fun d => admission d == .signatureRejected)).length = 2 := by
  decide +kernel

theorem admission_typed (p : TypedValue.Profile) :
    admission (profileDType p) = .typed p := by cases p <;> rfl

/-- The product stores the two existing semantic carriers. Dtype dispatch
chooses exactly one carrier; the theorem's result projection is branch-specific.
This does not require re-encoding the old twelve routes as binary64. -/
abbrev Runtime (α : Type) := FP.Structural.State α × FP64Source.Memory

def wideFinal (num : FP64Source.Numerics) (undef : Nat → FP64Source.F64)
    (order : Nat → List Nat) (x output : Tensor) (n grid block : Nat)
    (mem : FP64Source.Memory) : FP64Source.Memory :=
  (FP64Source.runSchedule num x.region output.region n block (order grid)
    (FP64Source.initial block mem (fun i => undef i.val))).mem

def logicalLaunch {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (cmp16 cmp32 : α → α → Bool) (num : FP64Source.Numerics)
    (undef : Nat → FP64Source.F64) (order : Nat → List Nat)
    (x output : Tensor) (n grid block : Nat) : Run (Runtime α) Unit :=
  fun s => match admission x.dtype with
    | .typed p => match runPrefix (TypedValue.backend M cmp16 cmp32)
        (TypedValue.source p x.region output.region n block) s.1 grid with
      | none => .error .logicalInterpreter s
      | some t => .ok () (t, s.2)
    | .fp64 => .ok () (s.1, wideFinal num undef order x output n grid block s.2)
    | .comparisonRejected => .error (.native "frontend: fp8 ordered comparison rejected") s
    | .signatureRejected => .error (.native "frontend: fp8 signature rejected") s
    | .outsideDomain => .error (.unsupportedLogicalProfile x.dtype) s

def withLogicalLaunch {α : Type} [Inhabited α] (P : Primitives (Runtime α))
    (M : FP.Structural.Algebra α) (cmp16 cmp32 : α → α → Bool)
    (num : FP64Source.Numerics) (undef : Nat → FP64Source.F64)
    (order : Nat → List Nat) : Primitives (Runtime α) :=
  { P with launch := logicalLaunch M cmp16 cmp32 num undef order }

/-- Launch success is derived from the existing typed source proof; no kernel
outcome is an assumption, and no existing implementation is rewritten. -/
theorem relu_typed_success {α : Type} [Inhabited α] (P : Primitives (Runtime α))
    (M : FP.Structural.Algebra α) (cmp16 cmp32 : α → α → Bool)
    (num : FP64Source.Numerics) (undef : Nat → FP64Source.F64)
    (order : Nat → List Nat) (p : TypedValue.Profile) (x output : Tensor)
    (s d a m : Runtime α) (trace : List Event) (device : String) (n : Nat)
    (dtype : x.dtype = profileDType p)
    (hd : P.device x s = .ok device d)
    (ha : P.emptyLike x "float32" device d = .ok output a)
    (hm : P.numel output a = .ok n m) :
    ∃ final, relu (withLogicalLaunch P M cmp16 cmp32 num undef order) x ⟨s, trace⟩ =
        .ok output ⟨final, trace ++ successTrace x output device n⟩ ∧
      final.1.mem = (if gridCount n 1024 = 0 then m.1.mem else
        TypedValue.reference M cmp16 cmp32 p.kind x.region output.region n 1024 m.1.mem) ∧
      final.2 = m.2 := by
  obtain ⟨t, ht, hmem⟩ := TypedValue.prefix_correct M cmp16 cmp32 p
    x.region output.region n 1024 m.1 (gridCount n 1024)
  refine ⟨(t, m.2), ?_, hmem, rfl⟩
  apply HostWrapper.relu_success
  · exact hd
  · exact ha
  · exact hm
  · simp [withLogicalLaunch, logicalLaunch, dtype, admission_typed, ht]

/-- Host-wrapper execution and return are connected to the actual fp64 grid.
The primitive hypotheses describe only device, allocation and numel outcomes.
All ceil-grid permutations produce the exact first-block formula and tail frame. -/
theorem relu_fp64_success {α : Type} [Inhabited α] (P : Primitives (Runtime α))
    (M : FP.Structural.Algebra α) (cmp16 cmp32 : α → α → Bool)
    (num : FP64Source.Numerics) (undef : Nat → FP64Source.F64)
    (order : Nat → List Nat) (x output : Tensor)
    (s d a m : Runtime α) (trace : List Event) (device : String) (n : Nat)
    (dtype : x.dtype = "float64")
    (complete : (order (gridCount n 1024)).Perm (List.range (gridCount n 1024)))
    (hd : P.device x s = .ok device d)
    (ha : P.emptyLike x "float32" device d = .ok output a)
    (hm : P.numel output a = .ok n m) :
    ∃ final, relu (withLogicalLaunch P M cmp16 cmp32 num undef order) x ⟨s, trace⟩ =
        .ok output ⟨final, trace ++ successTrace x output device n⟩ ∧
      final.2 = (if gridCount n 1024 = 0 then m.2 else
        FP64Source.reference num x.region output.region n 1024 m.2) ∧
      final.1 = m.1 := by
  refine ⟨(m.1, wideFinal num undef order x output n (gridCount n 1024) 1024 m.2), ?_, ?_, rfl⟩
  · apply HostWrapper.relu_success
    · exact hd
    · exact ha
    · exact hm
    · simp [withLogicalLaunch, logicalLaunch, dtype, admission, selectProfile]
  · exact FP64Source.grid_memory num x.region output.region n 1024
      (order (gridCount n 1024)) (FP64Source.initial 1024 m.2 (fun i => undef i.val)) complete

theorem relu_comparison_rejection {α : Type} [Inhabited α] (P : Primitives (Runtime α))
    (M : FP.Structural.Algebra α) (cmp16 cmp32 : α → α → Bool)
    (num : FP64Source.Numerics) (undef : Nat → FP64Source.F64)
    (order : Nat → List Nat) (x output : Tensor)
    (s d a m : Runtime α) (trace : List Event) (device : String) (n : Nat)
    (rejected : admission x.dtype = .comparisonRejected)
    (hd : P.device x s = .ok device d)
    (ha : P.emptyLike x "float32" device d = .ok output a)
    (hm : P.numel output a = .ok n m) :
    relu (withLogicalLaunch P M cmp16 cmp32 num undef order) x ⟨s, trace⟩ =
      .error (.native "frontend: fp8 ordered comparison rejected")
        ⟨m, trace ++ (successTrace x output device n).take 5⟩ := by
  apply HostWrapper.launch_error_stops_return
  · exact hd
  · exact ha
  · exact hm
  · simp [withLogicalLaunch, logicalLaunch, rejected]

theorem relu_signature_rejection {α : Type} [Inhabited α] (P : Primitives (Runtime α))
    (M : FP.Structural.Algebra α) (cmp16 cmp32 : α → α → Bool)
    (num : FP64Source.Numerics) (undef : Nat → FP64Source.F64)
    (order : Nat → List Nat) (x output : Tensor)
    (s d a m : Runtime α) (trace : List Event) (device : String) (n : Nat)
    (rejected : admission x.dtype = .signatureRejected)
    (hd : P.device x s = .ok device d)
    (ha : P.emptyLike x "float32" device d = .ok output a)
    (hm : P.numel output a = .ok n m) :
    relu (withLogicalLaunch P M cmp16 cmp32 num undef order) x ⟨s, trace⟩ =
      .error (.native "frontend: fp8 signature rejected")
        ⟨m, trace ++ (successTrace x output device n).take 5⟩ := by
  apply HostWrapper.launch_error_stops_return
  · exact hd
  · exact ha
  · exact hm
  · simp [withLogicalLaunch, logicalLaunch, rejected]

#axiomsClean finite_inventory
#axiomsClean admission_typed
#axiomsClean relu_typed_success
#axiomsClean relu_fp64_success
#axiomsClean relu_comparison_rejection
#axiomsClean relu_signature_rejection
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.FP64Host
