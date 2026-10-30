import bench.repairs.tritonbench_g.lightning_attention.Proofs.Grid
namespace VeriTile.Bench.Repairs.LightningAttention.Native
open VeriTile Triton FP.Structural
noncomputable section
abbrev Word := BitVec 32

/-- Legal finite WGMMA N shape, in multiples of eight through 256. -/
abbrev Width := Fin 32
def Width.columns (w : Width) : Nat := 8 * (w.val + 1)
theorem Width.positive (w : Width) : 0 < w.columns := by simp [Width.columns]

inductive Backend where
  | warp
  | warpGroup (width : Width)

/-- A static compiler layout choice depending only on logical matrix shape.
It cannot inspect data, a dot result, memory, or the kernel's output. -/
structure Plan where
  backend : Nat → Nat → Backend
  /-- Compiler scale-D choice for a fused syntactic literal-zero seed.
  This is a static shape choice; operand words are not inspected. -/
  zeroSeedLive : Nat → Nat → Bool

def observedBackend (m n : Nat) : Backend :=
  if m < 64 then .warp else .warpGroup
    (if n ≤ 16 then ⟨1,by decide⟩ else if n ≤ 32 then ⟨3,by decide⟩
     else if n ≤ 64 then ⟨7,by decide⟩ else if n ≤ 128 then ⟨15,by decide⟩
     else ⟨31,by decide⟩)

/-- The six retained examples instantiate this layout. Larger D/E profiles
must supply their actual compiler plan; this definition is not evidence about
unobserved specializations. -/
def observedPlan : Plan := ⟨observedBackend,fun _ _ => Bool.true⟩

/-- Exactly three local numerical families. Both tensor-core ports are a
single fixed K8 output component, with bounded position/layout parameters.
The WGMMA seed-enable flag is retained; mma.sync always takes its live seed.
No dot length, memory, source loop, or output tensor is a numerical parameter. -/
structure Instructions where
  add : Word → Word → Word
  wgmma8 : Width → Fin 64 → Fin 256 → (Fin 8 → Word) → (Fin 8 → Word) → Bool → Word → Word
  mma8 : Fin 16 → Fin 8 → (Fin 8 → Word) → (Fin 8 → Word) → Word → Word

structure LocalConformance (actual reference : Instructions) : Prop where
  add : ∀ a b, actual.add a b = reference.add a b
  wgmma8 : ∀ w r c a b live seed,
    actual.wgmma8 w r c a b live seed = reference.wgmma8 w r c a b live seed
  mma8 : ∀ r c a b seed, actual.mma8 r c a b seed = reference.mma8 r c a b seed

theorem instructions_eq (a b : Instructions) (h : LocalConformance a b) : a = b := by
  cases a; cases b
  have h1 := funext fun x => funext (h.add x)
  have h2 := funext fun w => funext fun r => funext fun c => funext fun x =>
    funext fun y => funext fun live => funext (h.wgmma8 w r c x y live)
  have h3 := funext fun r => funext fun c => funext fun x => funext fun y => funext (h.mma8 r c x y)
  congr

/-- The final arithmetic graph contains only literal words, scalar add and
bounded instruction nodes. A variable-length source dot is absent here. -/
inductive Graph where
  | bits : Word → Graph
  | add : Graph → Graph → Graph
  | wgmma8 : Width → Fin 64 → Fin 256 → (Fin 8 → Graph) → (Fin 8 → Graph) → Bool → Graph → Graph
  | mma8 : Fin 16 → Fin 8 → (Fin 8 → Graph) → (Fin 8 → Graph) → Graph → Graph

def Graph.eval (p : Instructions) : Graph → Word
  | .bits x => x
  | .add a b => p.add (a.eval p) (b.eval p)
  | .wgmma8 w r c a b live seed =>
    p.wgmma8 w r c (fun e => (a e).eval p) (fun e => (b e).eval p) live (seed.eval p)
  | .mma8 r c a b seed =>
    p.mma8 r c (fun e => (a e).eval p) (fun e => (b e).eval p) (seed.eval p)

def atOrZero (k : Nat) (x : Fin k → Graph) (i : Nat) : Graph :=
  if h : i < k then x ⟨i,h⟩ else .bits 0

def instruction (backend : Backend) (row col : Nat) (a b : Fin 8 → Graph)
    (live : Bool) (seed : Graph) : Graph :=
  match backend with
  | .warp => .mma8 ⟨row % 16,Nat.mod_lt _ (by decide)⟩
      ⟨col % 8,Nat.mod_lt _ (by decide)⟩ a b seed
  | .warpGroup w => .wgmma8 w ⟨row % 64,Nat.mod_lt _ (by decide)⟩
      ⟨col % w.columns, lt_of_lt_of_le (Nat.mod_lt _ w.positive) (by
        have := w.isLt; simp only [Width.columns]; omega)⟩ a b live seed

/-- A fully explicit consecutive-K8 instruction fold. The final partial
group is zero padded; admitted source K dimensions are divisible by eight.
The first unseeded WGMMA instruction disables scale-D. All later WGMMA
instructions, and every source fused-dot update, retain their live seeds. -/
def dotGraph (plan : Plan) (m k n row col : Nat) (a b : Fin k → Graph)
    (seeded : Bool) (seed : Graph) : Graph :=
  (List.range ((k+7)/8)).foldl (fun acc block =>
    instruction (plan.backend m n) row col (fun e => atOrZero k a (block*8+e.val))
      (fun e => atOrZero k b (block*8+e.val)) (seeded || decide (block ≠ 0)) acc) seed

theorem chunk_coverage (k i : Nat) (hi : i < k) :
    i / 8 < (k+7)/8 ∧ i % 8 < 8 ∧ i / 8 * 8 + i % 8 = i := by
  omega

theorem graph_conformance (actual reference : Instructions) (h : LocalConformance actual reference)
    (g : Graph) : g.eval actual = g.eval reference := by rw [instructions_eq actual reference h]

#axiomsClean instructions_eq
#axiomsClean chunk_coverage
#axiomsClean graph_conformance
#specNonCircular LocalConformance avoiding [Graph.eval,dotGraph]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LightningAttention.Native
