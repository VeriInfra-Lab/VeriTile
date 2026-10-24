import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullAlgorithm
open Lean VeriTile.Triton

/-- The source flag is constexpr: each compiled kernel contains its selected formula. -/
syntax "klOpsRealForward%" "[" ident "]" : term
set_option hygiene false in
macro_rules
  | `(klOpsRealForward%[$branch:ident]) => do
    let compute : TSyntax `tritonStmt ← if branch.getId == `logarithmic then
      `(tritonStmt| loss = tl.exp(y_true) * (y_true - y))
      else `(tritonStmt| loss = y_true * (tl.log(tl.maximum(y_true, $(eps))) - y))
    `(fun (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) =>
      ComputeKernel.mk [Y, T] [O] (triton {
        tl.region Y = tl.float32, T = tl.float32, O = tl.float32
        pid = tl.program_id(0).to(tl.int64)
        Y += pid * $(sy)
        T += pid * $(st)
        O += pid * $(so)
        base_offsets = tl.arange(0, $(B))
        loss_sum = 0.0
        for i in range(0, $((N : Int)), $(B)) {
          offsets = i + base_offsets
          mask = offsets < $((N : Int))
          y = tl.load(Y + offsets, mask=mask, other=0.0)
          y_true = tl.load(T + offsets, mask=mask, other=0.0)
          $compute
          if $(reduction) == $(0) {
            tl.store(O + offsets, loss, mask=mask)
          } else {
            loss_sum += tl.sum(loss, axis=0)
          }
        }
        if $(reduction) != $(0) {
          tl.store(O, loss_sum)
        }
      }).surfaceBody)

/-- Complete forward control flow for the ideal real projection. Storage casts and
precision are erased only on this real surface; the typed FP source is separate. -/
def forwardSurface (Y T O : RegionName) (sy st so N B : Nat)
    (eps : ℝ) (logTarget : Bool) (reduction : Nat) : ComputeKernel :=
  (if logTarget then klOpsRealForward%[logarithmic] Y T O sy st so N B eps reduction
  else klOpsRealForward%[defaultTarget] Y T O sy st so N B eps reduction).eraseDType

syntax "klOpsRealBackward%" "[" ident "]" : term
set_option hygiene false in
macro_rules
  | `(klOpsRealBackward%[$branch:ident]) => do
    let compute : TSyntax `tritonStmt ← if branch.getId == `logarithmic then
      `(tritonStmt| res = -tl.exp(target))
      else `(tritonStmt| res = target * -1)
    `(fun (T O : RegionName) (st so N B : Nat) =>
      ComputeKernel.mk [T] [O] (triton {
        tl.region T = tl.float32, O = tl.float32
        pid = tl.program_id(0).to(tl.int64)
        T += pid * $(st)
        O += pid * $(so)
        offsets = tl.arange(0, $(B))
        mask = offsets < $((N : Int))
        for i in range(0, $((N : Int)), $(B)) {
          offsets = i + tl.arange(0, $(B))
          mask = offsets < $((N : Int))
          target = tl.load(T + offsets, mask=mask, other=0.0)
          $compute
          tl.store(O + offsets, res, mask=mask)
        }
      }).surfaceBody)

/-- Complete backward control flow, retaining both source pre-loop assignments. -/
def backwardSurface (T O : RegionName) (st so N B : Nat)
    (logTarget : Bool) : ComputeKernel :=
  if logTarget then klOpsRealBackward%[logarithmic] T O st so N B
  else klOpsRealBackward%[defaultTarget] T O st so N B

def forward (Y T O : RegionName) (sy st so N B : Nat)
    (eps : ℝ) (logTarget : Bool) (reduction : Nat) : Kernel :=
  (forwardSurface Y T O sy st so N B eps logTarget reduction).toAlgKernel

def backward (T O : RegionName) (st so N B : Nat) (logTarget : Bool) : Kernel :=
  (backwardSurface T O st so N B logTarget).toAlgKernel

private theorem erased_source_supported (ck : ComputeKernel) (k : Kernel)
    (h : ck.toAlgorithm? = .ok k) :
    ck.eraseDType.toAlgorithm? = .ok ck.eraseDType.toAlgKernel := by
  simp [ComputeKernel.eraseDType, h]

theorem forward_supported (Y T O : RegionName) (sy st so N B : Nat)
    (eps : ℝ) (logTarget : Bool) (reduction : Nat) :
    (forwardSurface Y T O sy st so N B eps logTarget reduction).toAlgorithm? =
      .ok (forward Y T O sy st so N B eps logTarget reduction) := by
  cases logTarget <;>
    exact erased_source_supported _ _ rfl

theorem backward_supported (T O : RegionName) (st so N B : Nat) (logTarget : Bool) :
    (backwardSurface T O st so N B logTarget).toAlgorithm? =
      .ok (backward T O st so N B logTarget) := by
  cases logTarget <;> rfl

end VeriTile.Bench.TritonBenchG.KldivOps.FullAlgorithm

#guard_msgs (drop info) in
#auditModuleAxioms
