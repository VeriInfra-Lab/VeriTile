import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Direct_Workset

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Linearization
open Workset

/-- Ordinary per-address atomic linearization, expressed by local transitions.
The rules contain neither a whole-output equation nor numerical identities.
Different addresses may have different orders. -/
inductive Trace {α : Type} (step : Fin 64 → α → α) : List (Fin 64) → α → α → Prop
  | nil (value : α) : Trace step [] value value
  | cons (split : Fin 64) (rest : List (Fin 64)) (before middle after : α)
      (atomic : middle=step split before) (tail : Trace step rest middle after) :
      Trace step (split::rest) before after

theorem Trace.fold {α : Type} {step : Fin 64 → α → α} {order : List (Fin 64)}
    {before after : α} (trace : Trace step order before after) :
    after=order.foldl (fun old split => step split old) before := by
  induction trace with
  | nil => rfl
  | cons split rest before middle after atomic tail ih =>
    simpa only [List.foldl_cons,atomic] using ih

def events (h : Call) (orders : Orders h) (indices : Nat → Int)
    (b : Fin h.batches) (n : Fin h.N) : List (Fin 64) :=
  if indices b.val=-1 then [] else orders.order b n

/-- The local runtime boundary: every issued event linearizes once at its own
address. `Orders.complete` supplies the exact 64-event permutation. Empty
sentinel traces preserve their entry value. No program execution or successful
host operation is supplied by this structure. -/
structure Run {α : Type} (h : Call) (orders : Orders h) (indices : Nat → Int)
    (initial : Fin h.batches → Fin h.N → α)
    (step : Fin h.batches → Fin h.N → Fin 64 → α → α) where
  output : Fin h.batches → Fin h.N → α
  cells : ∀ b n,Trace (step b n) (events h orders indices b n) (initial b n) (output b n)

theorem Run.readback {α : Type} (h : Call) (orders : Orders h) (indices : Nat → Int)
    (initial : Fin h.batches → Fin h.N → α)
    (step : Fin h.batches → Fin h.N → Fin 64 → α → α)
    (run : Run h orders indices initial step) (b : Fin h.batches) (n : Fin h.N) :
    run.output b n=if indices b.val=-1 then initial b n else
      (orders.order b n).foldl (fun old split => step b n split old) (initial b n) := by
  have result := (run.cells b n).fold
  unfold events at result
  split_ifs at result ⊢ <;> simpa only [List.foldl_nil] using result

#axiomsClean Trace.fold
#axiomsClean Run.readback
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Linearization
