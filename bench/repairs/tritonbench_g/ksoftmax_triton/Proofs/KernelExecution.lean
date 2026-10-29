import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Host

/-! Kernel dispatch without wrapper checks, allocator, or a successful-run
assumption. Each row calls the interpreter of the complete typed source.
-/
namespace VeriTile.Bench.Repairs.Ksoftmax.KernelExecution
open VeriTile.Triton FP.Structural
noncomputable section

def source (op : Operation) (Q K D : Nat) : ComputeKernel :=
  match op with
  | .forward log causal mode => forward "dst" "left" "right" (Q*K) K (Q*K) K
      (if mode.present then K else 0) K D log mode causal Bool.false
  | .backward log causal => backward "dst" "left" "right" (Q*K) K (Q*K) K (Q*K) K K D log causal

def row {α : Type} [Inhabited α] (model : Algebra α) (op : Operation)
    (Q K D batch query : Nat) (s : State α) : Option (State α) :=
  run (FP.ProfiledRewrite.engine model) (source op Q K D).surfaceBody (atProgram s batch query)

def rows {α : Type} [Inhabited α] (model : Algebra α) (op : Operation)
    (B Q K D : Nat) : List (Fin B × Fin Q) → State α → Option (State α)
  | [],s => some s
  | p::ps,s => (row model op Q K D p.1 p.2 s).bind (rows model op B Q K D ps)

def exec {α : Type} [Inhabited α] (model : Algebra α) (depth : Host.DepthLibrary)
    (op : Operation) (x : TensorMetadata) (s : State α) : Option (State α) :=
  if x.numel=0 then some s else
    rows model op (batches x) (rowsOf x) (columns x) (depth.depth (columns x))
      (programs (batches x) (rowsOf x)) s
where rowsOf := VeriTile.Bench.Repairs.Ksoftmax.rows

theorem row_success {α : Type} [Inhabited α] (model : Algebra α) (op : Operation)
    (Q K D : Nat) (hD : 0<D) (batch query : Nat) (s : State α) :
    row model op Q K D batch query s=
      some (runRow model op "dst" "left" "right" Q K D hD batch query s) := by
  cases op with
  | forward log causal mode =>
    cases mode with
    | absent =>
      have execution := (Floating.forward_run model "dst" "left" "right"
        (Q*K) K (Q*K) K 0 K D hD log .absent causal (atProgram s batch query)).property
      cases log <;> cases causal <;> exact execution
    | qk => exact (Floating.forward_run model "dst" "left" "right"
        (Q*K) K (Q*K) K K K D hD log .qk causal (atProgram s batch query)).property
    | bk => exact (Floating.forward_run model "dst" "left" "right"
        (Q*K) K (Q*K) K K K D hD log .bk causal (atProgram s batch query)).property
  | backward log causal => exact (Floating.backward_run model "dst" "left" "right"
      (Q*K) K (Q*K) K (Q*K) K K D log causal (atProgram s batch query)).property

theorem rows_success {α : Type} [Inhabited α] (model : Algebra α) (op : Operation)
    (B Q K D : Nat) (hD : 0<D) (schedule : List (Fin B × Fin Q)) (s : State α) :
    rows model op B Q K D schedule s=
      some (runRows model op "dst" "left" "right" B Q K D hD schedule s) := by
  induction schedule generalizing s with
  | nil => rfl
  | cons p ps ih =>
    simp only [rows,row_success model op Q K D hD,Option.bind_some]
    exact ih _

/-- The historical value/frame proofs refer to precisely this source execution. -/
theorem success {α : Type} [Inhabited α] (model : Algebra α) (depth : Host.DepthLibrary)
    (op : Operation) (x : TensorMetadata) (s : State α) :
    exec model depth op x s=some (Host.body model depth op x s) := by
  unfold exec Host.body
  split
  · rfl
  · exact rows_success model op _ _ _ _ (depth.positive _) _ s

theorem empty {α : Type} [Inhabited α] (model : Algebra α) (depth : Host.DepthLibrary)
    (op : Operation) (x : TensorMetadata) (s : State α) (h : x.numel=0) :
    exec model depth op x s=some s := by simp [exec,h]

#axiomsClean row_success
#axiomsClean rows_success
#axiomsClean success
#axiomsClean empty
end
end VeriTile.Bench.Repairs.Ksoftmax.KernelExecution
