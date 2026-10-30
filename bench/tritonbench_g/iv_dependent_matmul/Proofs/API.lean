import bench.tritonbench_g.iv_dependent_matmul.Proofs.Execution

/-! Ordered wrapper API contract. The tensor library supplies device conversion,
random allocations and the fresh output allocation. Launch success is linked to
Execution.host by its caller; failures propagate at their actual stage. These
ordinary external API outcomes do not impose new index-domain restrictions. -/
namespace VeriTile.Bench.Repairs.IvDependentMatmul.API
open TritonBenchG.IvDependentMatmul.Whole

inductive Stage where
  | device | randomA | randomB | emptyC | launch | returned
  deriving DecidableEq

/-- `object` is the returned tensor identity; metadata is the library contract. -/
structure Tensor (metadata : TensorMetadata) where
  object : Nat

structure Calls (r : Request) (Device Error : Type) where
  device : Except Error Device
  randomA : Device → Except Error (Tensor (inputMetadata r))
  randomB : Device → Except Error (Tensor (weightMetadata r))
  emptyC : Device → Except Error (Tensor (outputMetadata r))
  launch : Device → Tensor (inputMetadata r) → Tensor (weightMetadata r) →
    Tensor (outputMetadata r) → Except Error Unit

def run {r : Request} {Device Error : Type} (c : Calls r Device Error) :
    Except (List Stage × Error) (List Stage × Tensor (outputMetadata r)) :=
  match c.device with
  | .error e => .error ([.device],e)
  | .ok d => match c.randomA d with
    | .error e => .error ([.device,.randomA],e)
    | .ok a => match c.randomB d with
      | .error e => .error ([.device,.randomA,.randomB],e)
      | .ok b => match c.emptyC d with
        | .error e => .error ([.device,.randomA,.randomB,.emptyC],e)
        | .ok out => match c.launch d a b out with
          | .error e => .error ([.device,.randomA,.randomB,.emptyC,.launch],e)
          | .ok _ => .ok ([.device,.randomA,.randomB,.emptyC,.launch,.returned],out)

theorem successful_return {r : Request} {Device Error : Type} (c : Calls r Device Error)
    (d : Device) (a : Tensor (inputMetadata r)) (b : Tensor (weightMetadata r))
    (out : Tensor (outputMetadata r)) (hd : c.device=.ok d) (ha : c.randomA d=.ok a)
    (hb : c.randomB d=.ok b) (hc : c.emptyC d=.ok out) (hl : c.launch d a b out=.ok ()) :
    run c=.ok ([.device,.randomA,.randomB,.emptyC,.launch,.returned],out) := by
  simp only [run,hd,ha,hb,hc,hl]

def FailurePrefix (trace : List Stage) : Prop :=
  trace=[.device] ∨ trace=[.device,.randomA] ∨ trace=[.device,.randomA,.randomB] ∨
  trace=[.device,.randomA,.randomB,.emptyC] ∨ trace=[.device,.randomA,.randomB,.emptyC,.launch]

theorem failure_prefix {r : Request} {Device Error : Type} (c : Calls r Device Error)
    (trace : List Stage) (error : Error) (h : run c=.error (trace,error)) : FailurePrefix trace := by
  unfold run at h
  split at h
  · cases h; exact Or.inl rfl
  · split at h
    · cases h; exact Or.inr (Or.inl rfl)
    · split at h
      · cases h; exact Or.inr (Or.inr (Or.inl rfl))
      · split at h
        · cases h; exact Or.inr (Or.inr (Or.inr (Or.inl rfl)))
        · split at h
          · cases h; exact Or.inr (Or.inr (Or.inr (Or.inr rfl)))
          · cases h

#axiomsClean successful_return
#axiomsClean failure_prefix
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.IvDependentMatmul.API
