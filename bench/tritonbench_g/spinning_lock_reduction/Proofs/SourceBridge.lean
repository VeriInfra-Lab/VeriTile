import bench.tritonbench_g.spinning_lock_reduction.Proofs.GeneratedSource

namespace VeriTile.Bench.TritonBenchG.SpinningLock.Frontend

/-- A readable compiler result. `compile_original` checks it against the ENTIRE
AST-generated source tree; the decoder below then evaluates these expressions.
It is not a hypothesis that this plan's execution equals the source result. -/
def originalPlan : Plan where
  start := 1
  stop := 10
  reducer := .eq (.srem .pid (.param .groupK)) (.literal 0)
  nextInitial := .add .pid (.literal 1)
  continues := .and (.lt .next (.add .pid (.param .groupK))) (.lt .next (.param .numSms))
  poll := ⟨.locks, [.next]⟩
  load := ⟨.partials, [.mul (.mul .next (.param .bm)) (.param .bn), .mul .row (.param .bn), .col]⟩
  advance := .add .next (.literal 1)
  publish := ⟨.partials, [.mul (.mul .pid (.param .bm)) (.param .bn), .mul .row (.param .bn), .col]⟩
  release := ⟨.locks, [.pid]⟩
  output := ⟨.output,
    [.mul (.add (.mul (.sdiv .pid (.param .numSms)) (.param .bm)) .row) (.param .strideM),
     .mul (.add (.mul (.srem .pid (.param .numSms)) (.param .bn)) .col) (.param .strideN)]⟩
  mask := .and
    (.lt (.add (.mul (.sdiv .pid (.param .numSms)) (.param .bm)) .row) (.param .m))
    (.lt (.add (.mul (.srem .pid (.param .numSms)) (.param .bn)) .col) (.param .n))

theorem compile_original : compile originalTypedSource = some originalPlan := by
  decide +kernel

@[simp] theorem reducer_expression (c : Config) (p : Nat) :
    evalB c p 0 0 originalPlan.reducer = c.reducer p := rfl
@[simp] theorem next_expression (c : Config) (p : Nat) :
    evalI c p 0 0 originalPlan.nextInitial = c.pid p + 1 := rfl
@[simp] theorem guard_expression (c : Config) (p : Nat) (n : I32) :
    evalB c p n 0 originalPlan.continues = c.continues p n := rfl
@[simp] theorem advance_expression (c : Config) (p : Nat) (n : I32) :
    evalI c p n 0 originalPlan.advance = n + 1 := rfl
@[simp] theorem mask_expression (c : Config) (p lane : Nat) :
    evalB c p 0 lane originalPlan.mask = c.outputMask p lane := rfl
@[simp] theorem poll_address (c : Config) (p : Nat) (n : I32) :
    address c p n 0 originalPlan.poll = n.toInt := by
  simp only [address, originalPlan, evalI, ↓List.map_cons, ↓List.map_nil,
    List.foldl_cons, List.foldl_nil, Int.zero_add]
@[simp] theorem release_address (c : Config) (p : Nat) :
    address c p 0 0 originalPlan.release = (c.pid p).toInt := by
  simp only [address, originalPlan, evalI, ↓List.map_cons, ↓List.map_nil,
    List.foldl_cons, List.foldl_nil, Int.zero_add]
private theorem address_two (c : Config) (p : Nat) (n : I32) (lane : Nat)
    (r : Region) (a b : IExpr) :
    address c p n lane ⟨r, [a,b]⟩ =
      (evalI c p n lane a).toInt + (evalI c p n lane b).toInt := by
  change (0 + (evalI c p n lane a).toInt) + (evalI c p n lane b).toInt = _
  rw [Int.zero_add]

private theorem address_three (c : Config) (p : Nat) (n : I32) (lane : Nat)
    (r : Region) (a b d : IExpr) :
    address c p n lane ⟨r, [a,b,d]⟩ =
      (evalI c p n lane a).toInt + (evalI c p n lane b).toInt + (evalI c p n lane d).toInt := by
  change ((0 + (evalI c p n lane a).toInt) + (evalI c p n lane b).toInt) +
    (evalI c p n lane d).toInt = _
  rw [Int.zero_add]

@[simp] theorem load_address (c : Config) (p : Nat) (n : I32) (lane : Nat) :
    address c p n lane originalPlan.load = c.pAddr n lane := by
  change address c p n lane ⟨.partials,
    [.mul (.mul .next (.param .bm)) (.param .bn), .mul .row (.param .bn), .col]⟩ = _
  rw [address_three]
  rfl
@[simp] theorem publish_address (c : Config) (p lane : Nat) :
    address c p 0 lane originalPlan.publish = c.pAddr (c.pid p) lane := by
  change address c p 0 lane ⟨.partials,
    [.mul (.mul .pid (.param .bm)) (.param .bn), .mul .row (.param .bn), .col]⟩ = _
  rw [address_three]
  rfl
@[simp] theorem output_address (c : Config) (p lane : Nat) :
    address c p 0 lane originalPlan.output = c.cAddr p lane := by
  change address c p 0 lane ⟨.output,
    [.mul (.add (.mul (.sdiv .pid (.param .numSms)) (.param .bm)) .row) (.param .strideM),
     .mul (.add (.mul (.srem .pid (.param .numSms)) (.param .bn)) .col) (.param .strideN)]⟩ = _
  rw [address_two]
  rfl

/-- Explicit scalar guards, signed address operations, masks, CAS old-word use,
and all phase transitions coincide for EVERY state and every configuration. -/
theorem decode_original (c : Config) (p : Nat) (s : State V) :
    decodePlan originalPlan c p s = decode c p s := by
  unfold decodePlan decode
  simp only [show originalPlan.stop = 10 from rfl, reducer_expression, next_expression,
    guard_expression, advance_expression, mask_expression, poll_address,
    release_address, load_address, publish_address, output_address]
  all_goals rfl


@[simp] theorem initial_original (b : Backend V) (mem : Memory V) :
    initialPlan originalPlan b mem = initial b mem := rfl

@[simp] theorem step_original (b : Backend V) (c : Config) (p : Nat) (s : State V) :
    stepPlan originalPlan b c p s = step b c p s := by
  simp only [stepPlan, step, decode_original]

/-- The bridge is lifted by induction over an arbitrary scheduler prefix;
no bounded unrolling of either source while is used. -/
theorem run_original (b : Backend V) (c : Config) (schedule : List Nat) (s : State V) :
    runPlan originalPlan b c schedule s = run b c schedule s := by
  induction schedule generalizing s with
  | nil => rfl
  | cons p ps ih => simp only [runPlan, run, step_original, ih]

/-- A rejected typed source produces none. Accepted source executes its compiled
expressions, not a separately supplied theorem-side final answer. -/
def compileAndRun (source : List Statement) (b : Backend V) (c : Config)
    (schedule : List Nat) (mem : Memory V) : Option (List Event × State V) :=
  (compile source).map fun plan => runPlan plan b c schedule (initialPlan plan b mem)

theorem original_source_execution (b : Backend V) (c : Config)
    (schedule : List Nat) (mem : Memory V) :
    compileAndRun originalTypedSource b c schedule mem =
      some (run b c schedule (initial b mem)) := by
  simp only [compileAndRun, compile_original, Option.map_some, initial_original, run_original]

/-- A source tree missing the entire loop fails admission. -/
theorem incomplete_source_rejected : compile (originalTypedSource.take 4) = none := by
  decide +kernel

/-- The compiler checks all statements rather than accepting a matching prefix. -/
theorem extra_source_instruction_rejected : compile (originalTypedSource ++ [.pass]) = none := by
  decide +kernel

#axiomsClean compile_original
#axiomsClean decode_original
#axiomsClean run_original
#axiomsClean original_source_execution
#axiomsClean incomplete_source_rejected
#axiomsClean extra_source_instruction_rejected
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SpinningLock.Frontend
