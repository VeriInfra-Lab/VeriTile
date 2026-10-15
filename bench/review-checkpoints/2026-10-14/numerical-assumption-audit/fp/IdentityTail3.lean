import bench.tritonbench_g.adam_update_triton.FPEquiv
import bench.tritonbench_g.add_example.FPEquiv
import bench.tritonbench_g.add_value.FPEquiv
import bench.tritonbench_g.apply_penalty.FPEquiv
import bench.tritonbench_g.attention_forward_triton.FPEquiv
import bench.tritonbench_g.attention_fwd_triton1.FPEquiv
import bench.tritonbench_g.attention_fwd_triton2.FPEquiv
import bench.tritonbench_g.attention_fwd_triton3.FPEquiv
import bench.tritonbench_g.attention_kernel.FPEquiv
import bench.tritonbench_g.attention_llama.FPEquiv
import bench.tritonbench_g.attention_score.FPEquiv
import bench.tritonbench_g.attn_fwd_causal.FPEquiv
import bench.tritonbench_g.attn_fwd_triton.FPEquiv
import bench.tritonbench_g.batched_vecmat_mult.FPEquiv
import bench.tritonbench_g.bgmv_shrink_kernel.FPEquiv
import bench.tritonbench_g.block_sparse_attn.FPEquiv
import bench.tritonbench_g.bmm_chunk_bwd.FPEquiv
import bench.tritonbench_g.bmm_chunk_fwd.FPEquiv
import bench.tritonbench_g.bmm_optimized.FPEquiv
import bench.tritonbench_g.cache_transform_triton.FPEquiv
import bench.tritonbench_g.chunk_bwd_dqkg.FPEquiv
import bench.tritonbench_g.chunk_cumsum_kernel.FPEquiv
import bench.tritonbench_g.chunk_cumsum_vector.FPEquiv
import bench.tritonbench_g.chunk_delta_fwd.FPEquiv
import bench.tritonbench_g.chunk_gate_recurrence.FPEquiv
import bench.tritonbench_g.chunk_gated_attention.FPEquiv
import bench.tritonbench_g.chunk_gla_fwd.FPEquiv
import bench.tritonbench_g.chunk_gla_simple.FPEquiv
import bench.tritonbench_g.chunk_linear_attn.FPEquiv
import bench.tritonbench_g.chunk_retention.FPEquiv
import bench.tritonbench_g.chunk_retention_ops.FPEquiv
import bench.tritonbench_g.chunked_cumsum_fwd.FPEquiv
import bench.tritonbench_g.context_attn_bloom.FPEquiv
import bench.tritonbench_g.context_attn_fwd.FPEquiv
import bench.tritonbench_g.context_attn_llama.FPEquiv
import bench.tritonbench_g.context_attn_mistral.FPEquiv
import bench.tritonbench_g.context_attn_nopad.FPEquiv
import bench.tritonbench_g.cosine_compute.FPEquiv
import bench.tritonbench_g.cross_entropy1.FPEquiv
import bench.tritonbench_g.cross_entropy2.FPEquiv
import bench.tritonbench_g.cross_entropy_ops.FPEquiv
import bench.tritonbench_g.decay_cumsum.FPEquiv
import bench.tritonbench_g.dequantize_matmul.FPEquiv
import bench.tritonbench_g.dequantize_rowwise.FPEquiv
import bench.tritonbench_g.destindex_copy.FPEquiv
import bench.tritonbench_g.destindex_copy_kv1.FPEquiv
import bench.tritonbench_g.destindex_copy_kv2.FPEquiv
import bench.tritonbench_g.diag_ssm_triton.FPEquiv
import bench.tritonbench_g.dropout_triton.FPEquiv
import bench.tritonbench_g.embedding_triton_kernel.FPEquiv
import bench.tritonbench_g.f8_conversion_utils.FPEquiv
import bench.tritonbench_g.fast_ce_loss.FPEquiv
import bench.tritonbench_g.fast_layernorm.FPEquiv
import bench.tritonbench_g.fast_rms_layernorm.FPEquiv
import bench.tritonbench_g.fast_rope_embedding.FPEquiv
import bench.tritonbench_g.flash_attn.FPEquiv
import bench.tritonbench_g.fused_activation.FPEquiv
import bench.tritonbench_g.geglu_tanh_triton.FPEquiv
import bench.tritonbench_g.index_select_bwd.FPEquiv
import bench.tritonbench_g.index_select_cat.FPEquiv
import bench.tritonbench_g.kldiv_compute.FPEquiv
import bench.tritonbench_g.kldiv_ops.FPEquiv
import bench.tritonbench_g.kldiv_triton.FPEquiv
import bench.tritonbench_g.l2_norm_bwd.FPEquiv
import bench.tritonbench_g.l2_norm_triton1.FPEquiv
import bench.tritonbench_g.l2_norm_triton2.FPEquiv
import bench.tritonbench_g.llama_ff_triton.FPEquiv
import bench.tritonbench_g.log_softmax.FPEquiv
import bench.tritonbench_g.masked_add_cuda.FPEquiv
import bench.tritonbench_g.matmul_persistent_triton.FPEquiv
import bench.tritonbench_g.matrix_transpose.FPEquiv
import bench.tritonbench_g.max_reduction.FPEquiv
import bench.tritonbench_g.mean_reduction.FPEquiv
import bench.tritonbench_g.mul_exponent_compensator.FPEquiv
import bench.tritonbench_g.nested_loops_processing.FPEquiv
import bench.tritonbench_g.pow_scalar_tensor.FPEquiv
import bench.tritonbench_g.relu_strided_buffer.FPEquiv
import bench.tritonbench_g.relu_triton_kernel.FPEquiv
import bench.tritonbench_g.rms_matmul_rbe.FPEquiv
import bench.tritonbench_g.rms_norm_triton.FPEquiv
import bench.tritonbench_g.rms_rbe_matmul.FPEquiv
import bench.tritonbench_g.rmsnorm_fused.FPEquiv
import bench.tritonbench_g.rmsnorm_fused_llama.FPEquiv
import bench.tritonbench_g.rmsnorm_implementation.FPEquiv
import bench.tritonbench_g.rmsnorm_triton.FPEquiv
import bench.tritonbench_g.rotary_emb.FPEquiv
import bench.tritonbench_g.seeded_dropout.FPEquiv
import bench.tritonbench_g.sin_computation.FPEquiv
import bench.tritonbench_g.sin_kernel.FPEquiv
import bench.tritonbench_g.softmax_flaggems.FPEquiv
import bench.tritonbench_g.softmax_optimize.FPEquiv
import bench.tritonbench_g.softmax_triton1.FPEquiv
import bench.tritonbench_g.softmax_triton2.FPEquiv
import bench.tritonbench_g.softmax_triton3.FPEquiv
import bench.tritonbench_g.square_matrix.FPEquiv
import bench.tritonbench_g.swiglu_backward.FPEquiv
import bench.tritonbench_g.swiglu_fwd.FPEquiv
import bench.tritonbench_g.swiglu_triton.FPEquiv
import bench.tritonbench_g.triton_mul2.FPEquiv
import bench.tritonbench_g.var_len_copy.FPEquiv
import bench.tritonbench_g.vector_addition.FPEquiv
import bench.tritonbench_g.vector_addition_custom.FPEquiv
import VeriTile.Meta.StatementAudit
open Lean Elab Command VeriTile.Meta
namespace FPInventoryProbe
private def fpProofType (name : Name) : Bool :=
  #[``VeriTile.Spec.Derivation, ``VeriTile.Spec.ProgramDerivation,
    ``VeriTile.Spec.FloatingPoint, ``VeriTile.Spec.ProgramSyntax.Derivation,
    ``VeriTile.Spec.ProgramSyntax.numerical,
    `VeriTile.Triton.FP.Equational.TermEq, `VeriTile.Triton.FP.Equational.OptionalEq,
    `VeriTile.Triton.FP.Structural.IO₁NumericalEquiv,
    `VeriTile.Triton.FP.Guarded.Equivalent,
    `VeriTile.Triton.FP.GuardedRewrite.Equivalent,
    `VeriTile.Triton.FP.Guarded.Equivalent₁ₓ₂,
    `VeriTile.Triton.FP.Scheduled.Equivalent₁ₓ₂,
    `VeriTile.Triton.FP.Scheduled.Equivalent₃,
    `VeriTile.Triton.FP.Scheduled.Equivalent₁,
    `VeriTile.Triton.FP.Scheduled.ConditionalEquivalent₁,
    `VeriTile.Triton.FP.SoftmaxShift.LibdeviceExpSub,
    `VeriTile.Triton.FP.LogSumExpShift.IntrinsicLogMul,
    `VeriTile.Triton.FP.LogSumExpShift.LogLibdeviceExp,
    `VeriTile.Triton.FP.WelfordInduction.CountConversion,
    `VeriTile.Triton.FP.Structural.CellRelated,
    `VeriTile.Triton.FP.Structural.ValueRelated].contains name

/-- Stop before unfolding either public relation. Follow aliases one step at
a time; a bounded failure is reported as an unwrapped legacy declaration. -/
private def specConclusion (e : Expr) : MetaM Expr := do
  let mut e := e
  for _ in [:64] do
    if e.isAppOf ``VeriTile.Spec.Real ||
        (match e.getAppFn with | .const name _ => fpProofType name | _ => Bool.false) then
      return e
    match ← Meta.unfoldDefinition? e (ignoreTransparency := Bool.true) with
    | some next => e := next
    | none => return e
  return e

private def specString? (projection : Name) (value : Expr) : MetaM (Option String) := do
  let field ← Meta.mkAppM projection #[value]
  return Meta.getStringValue? (← Meta.withTransparency .all <| Meta.whnf field)

private def shortSpecName : Name → String
  | .str _ name => name
  | name => name.toString

/-- Reader-facing names can differ from frozen experiment identifiers.
This affects only the concise display; contracts and detailed reports retain
their original identity. -/
private def fpAtomName (id : String) : String :=
  match id with
  | "EXP-SUB-INTRINSIC" => "exp_sub(tl.exp)"
  | "EXP-SUB" => "exp_sub(libdevice.exp)"
  | "EXP-ZERO" => "exp_zero(tl.exp)"
  | "EXP-ZERO-LIBDEVICE" => "exp_zero(libdevice.exp)"
  | "EXP-NEG-INF-SUB" => "exp_neg_inf_sub(tl.exp)"
  | "EXP-NEG-INF-SUB-LIBDEVICE" => "exp_neg_inf_sub(libdevice.exp)"
  | "LOG-MUL" => "log_mul(tl.log)"
  | "LOG-MUL-LIBDEVICE" => "log_mul(libdevice.log)"
  | "LOG-MUL-GUARDED" => "log_mul_split(libdevice.log)"
  | "LOG-MUL-GUARDED-INTRINSIC" => "log_mul_split(tl.log)"
  | "LOG-EXP" => "log_exp(tl.log, tl.exp)"
  | "LOG-EXP-LOG-LIBDEVICE" => "log_exp(libdevice.log, tl.exp)"
  | "LOG-EXP-LIBDEVICE" => "log_exp(tl.log, libdevice.exp)"
  | "LOG-EXP-FULL-LIBDEVICE" => "log_exp(libdevice.log, libdevice.exp)"
  | "LOG-EXP-GUARDED" => "log_exp_cancel(libdevice.log, libdevice.exp)"
  | "LOG-EXP-GUARDED-INTRINSIC" => "log_exp_cancel(tl.log, libdevice.exp)"
  | "LOG-EXP-GUARDED-FULL-INTRINSIC" => "log_exp_cancel(tl.log, tl.exp)"
  | "LOG-EXP-GUARDED-EXP-INTRINSIC" => "log_exp_cancel(libdevice.log, tl.exp)"
  | _ => id.toLower.replace "-" "_"

private def printFPAtom (entry : Expr) (_details : Bool := Bool.false) : MetaM Unit := do
  let rule ← Meta.mkAppM ``VeriTile.Spec.RuleEntry.rule #[entry]
  let contract ← Meta.mkAppM ``VeriTile.Spec.AtomicRule.contract #[rule]
  let id ← specString? ``VeriTile.Spec.Contract.ruleID contract
  let key ← specString? ``VeriTile.Spec.Contract.instanceKey contract
  let item := Lean.Json.mkObj [
    ("rule_id", id.map Lean.Json.str |>.getD .null),
    ("instance_key", key.map Lean.Json.str |>.getD .null)]
  logInfo m!"ATOM_JSON {item.compress}"

private structure FPAssumptionState where
  seen : Std.HashSet Expr := {}
  internalProofVars : Std.HashSet Expr := {}
  forwardedAtoms : Std.HashSet Expr := {}
  entries : Array Expr := #[]
  printedAdmissions : Array (Expr × Expr) := #[]
  unresolved : Bool := Bool.false
  fpConstants : Std.HashMap Name Bool := {}
  remaining : Nat := 100000

/-- Conservative reachability, including declaration types and projection
functions. A closed dependency graph without an FP proof type cannot hide an
atomic derivation. Cache negative results only after the entire graph has been
searched; mutual recursion must not turn a back-edge into a Bool.false negative. -/
private partial def fpConstantReachable (root : Name) (work : List Name)
    (seen : NameSet := {}) : StateRefT FPAssumptionState MetaM Bool := do
  match work with
  | [] =>
    modify fun s => { s with fpConstants := seen.foldl (fun m n => m.insert n Bool.false) s.fpConstants }
    return Bool.false
  | name :: rest =>
    if seen.contains name then return ← fpConstantReachable root rest seen
    if fpProofType name || (← get).fpConstants[name]? == some Bool.true then
      modify fun s => { s with fpConstants := s.fpConstants.insert root Bool.true }
      return Bool.true
    if (← get).fpConstants[name]? == some Bool.false then
      return ← fpConstantReachable root rest seen
    let seen := seen.insert name
    let env ← getEnv
    if isCoreConst env name then return ← fpConstantReachable root rest seen
    let some info := env.find? name | return Bool.true
    let mut next ← specExprConsts env info.type
    if let some value := info.value? then next := next ++ (← specExprConsts env value)
    fpConstantReachable root (next.toList ++ rest) seen

private def fpExprReachable (proof : Expr) : StateRefT FPAssumptionState MetaM Bool := do
  let env ← getEnv
  let mut names ← specExprConsts env proof
  -- Actual proof arguments can introduce FP dependencies into otherwise
  -- generic helpers. Include free-variable types instead of inspecting only
  -- the helper's declaration or the result type of this proof.
  let visit : StateRefT (Bool × Array Expr) MetaM Unit := proof.forEach fun e => do
    if e.isFVar then modify fun s => (s.1, s.2.push e)
    if e.isMVar then modify fun s => (Bool.true, s.2)
  let (_, localState) ← visit.run (Bool.false, #[])
  if localState.1 then return Bool.true
  for localProof in localState.2 do
    let type ← Meta.inferType localProof
    if type.getAppFn.isFVar || type.isMVar then return Bool.true
    names := names ++ (← specExprConsts env type)
  for name in names do
    if ← fpConstantReachable name [name] then return Bool.true
  return Bool.false

/-- Expose a proof's callable head using only beta/zeta reduction. One
`headBeta` pass can expose another applied lambda (notably the induction
combinator generated by `induction ... using`). Its proof arguments must not
be mistaken for unused data arguments while that lambda remains at the head. -/
private partial def exposeFPProofHead (proof : Expr) : Expr :=
  let reduced := proof.headBeta
  if reduced != proof then exposeFPProofHead reduced
  else
    match proof.getAppFn with
    | .mdata _ inner => exposeFPProofHead (mkAppN inner proof.getAppArgs)
    | .letE _ _ value body _ =>
        exposeFPProofHead (mkAppN (body.instantiate1 value) proof.getAppArgs)
    | _ => proof

/-- Walk instantiated proof bodies, not theorem types or the declared table.
We inspect all reachable proof branches, not a minimal logical dependency set.
An opaque derivation is reported explicitly rather than guessed from its table.
Printing within each local context also supports atoms beneath proof lambdas. -/
private partial def visitFPAssumptions (proof : Expr) :
    StateRefT FPAssumptionState MetaM Unit := do
  let proof := exposeFPProofHead proof
  if (← get).seen.contains proof then return
  if (← get).remaining == 0 then
    throwError "FP assumption inspection exceeded its proof traversal limit"
  modify fun s => { s with seen := s.seen.insert proof, remaining := s.remaining - 1 }
  match proof with
  | .mdata _ inner => visitFPAssumptions inner
  | .letE _ _ value body _ => visitFPAssumptions (body.instantiate1 value)
  | .lam .. => Meta.lambdaTelescope proof fun params body => do
      -- These binders belong to a proof body (e.g. recursion hypotheses).
      -- The headline's external premises have already been instantiated by
      -- the command's outer telescope and must still be reported if opaque.
      let previous := (← get).internalProofVars
      modify fun s => { s with internalProofVars := params.foldl (·.insert ·) previous }
      visitFPAssumptions body
      modify fun s => { s with internalProofVars := previous }
  | _ =>
    unless ← Meta.isProof proof do return
    unless ← fpExprReachable proof do return
    let args := proof.getAppArgs
    if proof.isAppOf ``VeriTile.Spec.Derivation.atom then
      let entry := args[2]!
      -- Track each syntax instantiation, but print the same concrete admission
      -- once when it is applied at several sites. Different experimental
      -- contracts/evidence stay distinct even when their textual IDs coincide.
      let normalized ← Meta.withTransparency .all <| Meta.whnf entry
      -- Only atoms forwarded by Derivation induction are covered by its
      -- inspected input proof. Other symbolic entries must remain visible.
      if (← get).forwardedAtoms.contains normalized then return
      unless (← get).entries.contains normalized do
        modify fun s => { s with entries := s.entries.push normalized }
        let rule ← Meta.mkAppM ``VeriTile.Spec.RuleEntry.rule #[entry]
        let contract ← Meta.mkAppM ``VeriTile.Spec.AtomicRule.contract #[rule]
        let evidence ← Meta.mkAppM ``VeriTile.Spec.RuleEntry.evidence #[entry]
        let concrete := (← specString? ``VeriTile.Spec.Contract.ruleID contract).isSome
        let identity := (← Meta.withTransparency .all <| Meta.whnf contract,
          ← Meta.withTransparency .all <| Meta.whnf evidence)
        -- Weak-head reduction may leave record projections inside fields.
        -- Compare closed admissions definitionally, so record updates that
        -- only rename registers do not produce duplicate printed assumptions.
        let closed := !identity.1.hasFVar && !identity.1.hasMVar &&
          !identity.2.hasFVar && !identity.2.hasMVar
        let duplicate ← if concrete && closed then
          (← get).printedAdmissions.anyM fun previous =>
            Meta.withTransparency .all do
              if previous.1.hasFVar || previous.1.hasMVar ||
                  previous.2.hasFVar || previous.2.hasMVar then return Bool.false
              return (← Meta.isDefEq previous.1 identity.1) &&
                (← Meta.isDefEq previous.2 identity.2)
          else pure Bool.false
        unless duplicate do
          modify fun s => { s with printedAdmissions := s.printedAdmissions.push identity }
          printFPAtom entry
      return
    if proof.isAppOf ``VeriTile.Spec.Derivation.rec then
      for i in [:args.size] do
        let arg := args[i]!
        -- Parameters and motive precede the refl and atom minor premises.
        -- The atom branch re-emits an atom from the input derivation, whose
        -- actual proof is also visited below. Do not generalize this to data
        -- recursors: unpacking an entry there can introduce a new assumption.
        if i == 4 then
          Meta.lambdaTelescope arg fun params body => do
            let previousVars := (← get).internalProofVars
            let previousAtoms := (← get).forwardedAtoms
            let forwarded := match params[0]? with
              | some entry => previousAtoms.insert entry
              | none => previousAtoms
            modify fun s => { s with
              internalProofVars := params.foldl (·.insert ·) previousVars
              forwardedAtoms := forwarded }
            visitFPAssumptions body
            modify fun s => { s with
              internalProofVars := previousVars
              forwardedAtoms := previousAtoms }
        else if ← Meta.isProof arg then visitFPAssumptions arg
      return
    let env ← getEnv
    if let .const name levels := proof.getAppFn then
      unless isCoreConst env name do
        let info ← getConstInfo name
        if let some value := info.value? then
          let body := value.instantiateLevelParams info.levelParams levels
          visitFPAssumptions (body.beta args)
          return
    if let .proj .. := proof.getAppFn then
      if let some reduced ← Meta.withTransparency .all <| Meta.reduceProj? proof.getAppFn then
        visitFPAssumptions (mkAppN reduced args)
        return
    -- Constructors, transports and library combinators retain their proof
    -- arguments. Skip data arguments, including unused rule-table entries.
    for arg in args do
      if ← Meta.isProof arg then visitFPAssumptions arg
    -- A local/opaque FP proof may conceal further atoms. Never print "none"
    -- in this case. Core recursors have their branches inspected above.
    let opaqueHead := match proof.getAppFn with
      | .fvar .. | .proj .. => Bool.true
      | .const name _ => match env.find? name with
          | some (.axiomInfo _) | some (.opaqueInfo _) => Bool.true
          | _ => Bool.false
      | _ => Bool.false
    let rec projectionRoot : Expr → Expr
      | .proj _ _ base => projectionRoot base.getAppFn
      | head => head
    let root := projectionRoot proof.getAppFn
    if let .proj _ _ base := proof.getAppFn then
      -- A primitive-law record can be consumed only through a projected
      -- equality. Inspect its record premise as well as the equality's type,
      -- which by itself no longer identifies the pending numerical law.
      visitFPAssumptions base
    if opaqueHead && !(← get).internalProofVars.contains root then
      let type ← specConclusion (← Meta.inferType proof)
      if (match type.getAppFn with | .const name _ => fpProofType name | _ => Bool.false) then
        modify fun s => { s with unresolved := Bool.true }
        logInfo m!"  unresolved FP proof: {← Meta.ppExpr proof} (atomic assumptions unavailable)"

/-- Print only the atomic assumptions referenced by a floating-point proof.
Works on ordinary theorems as well as registered `specification` headlines. -/
elab "#print_fp_identity " id:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverload id
  let info ← liftCoreM <| getConstInfo name
  -- Scalar-derived recurrence proofs can contain tens of thousands of
  -- instantiated nodes. Keep inspection bounded at ordinary call sites too.
  -- Core stores heartbeats in units 1000 times the public option value.
  liftTermElabM <| withTheReader Core.Context (fun ctx =>
      { ctx with maxHeartbeats :=
          if ctx.maxHeartbeats == 0 then 0 else max ctx.maxHeartbeats 4000000000 }) <|
      Meta.forallTelescope info.type fun params conclusion => do
    let target ← specConclusion conclusion
    unless (match target.getAppFn with | .const name _ => fpProofType name | _ => Bool.false) do
      throwError "{name}: expected a floating-point equivalence or derivation"
    logInfo m!"DECL {name}:"
    let proof := mkAppN (mkConst name (info.levelParams.map Level.param)) params
    let (_, state) ← (visitFPAssumptions proof).run {}
    if state.entries.isEmpty && !state.unresolved then logInfo "  none"


end FPInventoryProbe
#print_fp_identity VeriTile.Bench.TritonBenchG.PowScalarTensor.FPEquiv.integer_source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Equivalence.one_tile_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Equivalence.grid_stride_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.ReluTritonKernel.Equivalence.relu_fused_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.RmsMatmulRbe.FPEquiv.source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.RmsMatmulRbe.FPEquiv.qkv_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.RmsNormTriton.FPEquiv.source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.RmsRbeMatmul.FPEquiv.source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.RmsnormFused.FPEquiv.source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.FPEquiv.source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.RmsnormImplementation.FPEquiv.source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.RmsnormTriton.FPEquiv.source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.RotaryEmb.FPEquiv.source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.SeededDropout.Equivalence.seeded_dropout_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.SinComputation.FPEquiv.source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.SinKernel.FPEquiv.source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.Equivalence.backward_inner_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.Equivalence.backward_non_inner_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.Equivalence.forward_inner_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.Equivalence.forward_non_inner_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Equivalence.full_source_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.SoftmaxTriton1.FPEquiv.softmax_reciprocal_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.SoftmaxTriton2.FPEquiv.softmax_reciprocal_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.SoftmaxTriton3.FPEquiv.softmax_reciprocal_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.SquareMatrix.FPEquiv.square_index_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.SwigluBackward.FPEquiv.source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.SwigluFwd.FPEquiv.source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.SwigluTriton.FPEquiv.backward_index_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.SwigluTriton.FPEquiv.forward_index_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.TritonMul2.FPEquiv.mul2_commuted_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.TritonMul2.FPEquiv.mul2_inplace_commuted_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.VarLenCopy.FPEquiv.discrete_source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.VarLenCopy.FPEquiv.source_equivalence
#print_fp_identity VeriTile.Bench.TritonBenchG.VectorAddition.FPEquiv.add_commuted_equiv
#print_fp_identity VeriTile.Bench.TritonBenchG.VectorAdditionCustom.FPEquiv.add_commuted_equiv
