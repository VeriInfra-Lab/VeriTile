import bench.tritonbench_g.chunk_gate_recurrence.Proofs.PublicIO
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelCarrier
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! The sum-left source contracts the product on the right of its addition.
This changes no instruction or operand order: the same FMA ports receive
carry, gate and incoming. The proof covers every raw input word and does not
use the historical finite-site ADD-COMMUTE interface. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.ProgramPair
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelCarrier
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Native contraction of the existing candidate's right-hand product. -/
def rightAdd (isa : Instructions) (swap : Bool) (a b : Node) : Node :=
  match b.factors with
  | some (x,y) => atom (if swap then isa.fmaRN y x a.value else isa.fmaRN x y a.value)
  | none => atom (isa.addRN a.value b.value)

def candidateAlgebra (isa : Instructions) (plan : Reduction.ProductPlan 1024)
    (swap : Bool) : Algebra Node :=
  { algebra isa plan swap with
    binary := fun _ _ op a b => match op with
      | .mul => multiply isa a b
      | .add => rightAdd isa swap a b
      | _ => atom 0 }

def candidateSource (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) : ComputeKernel :=
  match kind with
  | .forward => Kernels.forwardSumLeft "S" "D" "O" "Seed" c.H c.N c.K c.V s.batch s.head s.key s.value seed
  | .backward => Kernels.backwardSumLeft "S" "D" "DI" "DG" "DL" "DS" c.H c.N c.K c.V 64 16

/-- Both syntax orders select the identical FMA on every pair of raw operands.
In particular no split-multiply/add relation or finite-value restriction enters. -/
theorem contracted_update (isa : Instructions) (swap : Bool) (a b x : Node) :
    add isa swap (multiply isa a b) x = rightAdd isa swap x (multiply isa a b) := rfl

def rightUpdate (carry incoming : RegName) : Op .real [64,16] :=
  .add .real (.consSame (.consSame .nil)) (KernelOps.matrix incoming)
    (.mul .real .scalarR (KernelOps.matrix carry) (KernelOps.scalar "d_i"))

theorem eval_update (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (carry incoming : RegName) (u : State Node) :
    evalOp (engine (algebra isa plan swap)) (some .fp32) (KernelOps.update carry incoming) u =
      evalOp (engine (candidateAlgebra isa plan swap)) (some .fp32) (rightUpdate carry incoming) u := by
  cases hc : u.regs .real [64,16] carry <;>
    cases hx : u.regs .real [64,16] incoming <;>
    cases hg : u.regs .real [] "d_i" <;>
    simp [KernelOps.update,rightUpdate,KernelOps.matrix,KernelOps.scalar,evalOp_unfold,hc,hx,hg,
      engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,bop,
      algebra,candidateAlgebra,multiply,add,rightAdd,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rfl

def carry : Kind → RegName | .forward => "acc" | .backward => "Dacc"
def incoming : Kind → RegName | .forward => "S_i" | .backward => "DS_i"
def bodyBefore (kind : Kind) (c : Shape) : List ComputeStmt :=
  Source.prefixBefore (match kind with | .forward => KernelOps.forwardBody c | .backward => KernelOps.backwardBody c) (carry kind)
def bodyAfter (kind : Kind) (c : Shape) : List ComputeStmt :=
  (match kind with | .forward => KernelOps.forwardBody c | .backward => KernelOps.backwardBody c).drop
    ((bodyBefore kind c).length+1)
def candidateParts (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (candidateSource kind c s seed).surfaceBody "_i").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 1,[],[]⟩

theorem body_original (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) :
    body kind c s seed = bodyBefore kind c ++
      [KernelOps.assign32 (carry kind) (KernelOps.update (carry kind) (incoming kind))] ++ bodyAfter kind c := by
  cases kind <;> rfl

theorem body_candidate (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) :
    (candidateParts kind c s seed).body = bodyBefore kind c ++
      [KernelOps.assign32 (carry kind) (rightUpdate (carry kind) (incoming kind))] ++ bodyAfter kind c := by
  cases kind <;> rfl

theorem body_before_run (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (kind : Kind) (c : Shape) (u : State Node) :
    run (engine (algebra isa plan swap)) (bodyBefore kind c) u =
      run (engine (candidateAlgebra isa plan swap)) (bodyBefore kind c) u := by
  cases kind <;> simp [bodyBefore,bodyAfter,carry,Source.prefixBefore,KernelOps.forwardBody,KernelOps.backwardBody,
    KernelOps.loadScalar,KernelOps.loadMatrix,KernelOps.assign32,KernelOps.storeMatrix,KernelOps.storeScalar,
    KernelOps.advance,KernelOps.retreat,KernelOps.gradient,KernelOps.matrixPtr,KernelOps.scalarPtr,
    KernelOps.matrix,KernelOps.scalar,KernelOps.stride,KernelOps.tileStride,
    run,FP.Structural.step,FP.Structural.store,evalExpr,evalComputeOp,evalOp_unfold,engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,
    algebra,candidateAlgebra]

theorem body_after_run (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (kind : Kind) (c : Shape) (u : State Node) :
    run (engine (algebra isa plan swap)) (bodyAfter kind c) u =
      run (engine (candidateAlgebra isa plan swap)) (bodyAfter kind c) u := by
  cases kind <;> simp [bodyBefore,bodyAfter,carry,Source.prefixBefore,KernelOps.forwardBody,KernelOps.backwardBody,
    KernelOps.loadScalar,KernelOps.loadMatrix,KernelOps.assign32,KernelOps.storeMatrix,KernelOps.storeScalar,
    KernelOps.advance,KernelOps.retreat,KernelOps.gradient,KernelOps.matrixPtr,KernelOps.scalarPtr,
    KernelOps.matrix,KernelOps.scalar,KernelOps.stride,KernelOps.tileStride,
    run,FP.Structural.step,FP.Structural.store,evalExpr,evalComputeOp,evalOp_unfold,engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,
    algebra,candidateAlgebra]

theorem body_run (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (u : State Node) :
    run (engine (algebra isa plan swap)) (body kind c s seed) u =
      run (engine (candidateAlgebra isa plan swap)) (candidateParts kind c s seed).body u := by
  have hp : run (engine (algebra isa plan swap)) (bodyBefore kind c) =
      run (engine (candidateAlgebra isa plan swap)) (bodyBefore kind c) :=
    funext (body_before_run isa plan swap kind c)
  have ht : run (engine (algebra isa plan swap)) (bodyAfter kind c) =
      run (engine (candidateAlgebra isa plan swap)) (bodyAfter kind c) :=
    funext (body_after_run isa plan swap kind c)
  have ha : run (engine (algebra isa plan swap))
      [KernelOps.assign32 (carry kind) (KernelOps.update (carry kind) (incoming kind))] =
      run (engine (candidateAlgebra isa plan swap))
      [KernelOps.assign32 (carry kind) (rightUpdate (carry kind) (incoming kind))] := by
    funext v
    simp only [run,FP.Structural.step,KernelOps.assign32,evalExpr,evalComputeOp]
    exact congrArg (fun (e : Option (Values Node .real [64,16])) =>
      (e.bind (fun values => some (v.setReg (carry kind) .real [64,16] values))).bind some)
      (ProgramPair.eval_update isa plan swap (carry kind) (incoming kind) v)
  simp only [body_original,body_candidate,run_append,hp,ht,ha]

theorem candidate_before (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) :
    (candidateParts kind c s seed).before = before kind c s seed := by
  cases kind <;> rfl

theorem before_run (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (u : State Node) :
    run (engine (algebra isa plan swap)) (before kind c s seed) u =
      run (engine (candidateAlgebra isa plan swap)) (before kind c s seed) u := by
  cases kind with
  | forward =>
    rw [(KernelPrefix.forward (algebra isa plan swap) c s seed u).property,
      (KernelPrefix.forward (candidateAlgebra isa plan swap) c s seed u).property]
    cases seed <;> rfl
  | backward =>
    rw [(KernelPrefix.backward (algebra isa plan swap) c s seed u).property,
      (KernelPrefix.backward (candidateAlgebra isa plan swap) c s seed u).property]
    rfl

def tailBefore (c : Shape) (s : SeedStrides) (seed : Bool) : List ComputeStmt :=
  Source.prefixBefore (after .backward c s seed) "Dacc"
def tailAfter (c : Shape) (s : SeedStrides) (seed : Bool) : List ComputeStmt :=
  (after .backward c s seed).drop ((tailBefore c s seed).length+1)

theorem tail_original (c : Shape) (s : SeedStrides) (seed : Bool) :
    after .backward c s seed = tailBefore c s seed ++
      [KernelOps.assign32 "Dacc" (KernelOps.update "Dacc" "DS_i")] ++ tailAfter c s seed := rfl

theorem tail_candidate (c : Shape) (s : SeedStrides) (seed : Bool) :
    (candidateParts .backward c s seed).after = tailBefore c s seed ++
      [KernelOps.assign32 "Dacc" (rightUpdate "Dacc" "DS_i")] ++ tailAfter c s seed := rfl

theorem tail_before_run (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (c : Shape) (s : SeedStrides) (seed : Bool) (u : State Node) :
    run (engine (algebra isa plan swap)) (tailBefore c s seed) u =
      run (engine (candidateAlgebra isa plan swap)) (tailBefore c s seed) u := by
  simp only [tailBefore,after,parts,source,Kernels.backward,ComputeKernel.surfaceBody,
    Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,↓reduceIte]
  simp [run,FP.Structural.step,FP.Structural.store,evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,algebra,candidateAlgebra]

theorem tail_after_run (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (c : Shape) (s : SeedStrides) (seed : Bool) (u : State Node) :
    run (engine (algebra isa plan swap)) (tailAfter c s seed) u =
      run (engine (candidateAlgebra isa plan swap)) (tailAfter c s seed) u := by
  simp only [tailAfter,tailBefore,after,parts,source,Kernels.backward,ComputeKernel.surfaceBody,
    Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,String.reduceEq,
    beq_self_eq_true,↓reduceIte]
  simp [run,FP.Structural.step,FP.Structural.store,evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,algebra,candidateAlgebra]

theorem range_run (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (stop stride : Nat) :
    ∀ start u, range (engine (algebra isa plan swap)) "_i" start stop stride (body kind c s seed) u =
      range (engine (candidateAlgebra isa plan swap)) "_i" start stop stride
        (candidateParts kind c s seed).body u := by
  intro start
  induction start using (measure (fun i => stop-i)).wf.induction with
  | h cur ih =>
    intro u
    rw [range,range]
    by_cases hz : stride=0
    · simp [hz]
    by_cases hc : cur<stop
    · simp only [hz,hc,if_false,if_true]
      rw [body_run]
      cases he : run (engine (candidateAlgebra isa plan swap))
        (candidateParts kind c s seed).body (u.setReg "_i" .nat [] (fun _ => cur)) with
      | none => rfl
      | some v =>
        simp only [bind,Option.bind_some]
        exact ih (cur+stride) (by simp_wf; omega) v
    · simp [hz,hc]

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.ProgramPair
