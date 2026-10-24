import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.KernelNativeTrees
import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.HostParameters

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelNative
open VeriTile.Triton FP.Structural Native
noncomputable section

/-- The compiler selector sees node kinds and dimensions only. Neither a raw
word nor a real literal can affect any routing, orientation, or tree choice. -/
inductive Form where
  | bits | literal | exponential | materialize
  | binary (op : Binary) (left right : Form)
  | reduction (op : Binary) (width : Nat)
  | dot (width row column : Nat)

def form : Expr → Form
  | .bits _ => .bits
  | .literal _ => .literal
  | .binary op a b => .binary op (form a) (form b)
  | .exponential _ => .exponential
  | .materialize _ => .materialize
  | .reduction op n _ => .reduction op n
  | .pendingDot d r c _ _ => .dot d r c

/-- Actual operand orientation in the saved SM80 PTX. Alpha times old mass
is the fused product; beta times tile mass is its rounded addend. -/
def reverse : Form → Bool
  | .binary .mul (.binary .add .literal (.dot ..)) .literal => Bool.true
  | .binary .mul .exponential .literal => Bool.true
  | .binary .mul .exponential (.binary .add ..) => Bool.true
  | .binary .mul .exponential (.reduction ..) => Bool.true
  | .binary .mul (.binary .div ..) .exponential => Bool.true
  | _ => Bool.false

/-- Owner records the logical column, which fixes the reduction replica.
MMA A fragments and reduction inputs are fetched from their own producer. -/
def route : Edge → Owner → Owner
  | .dotLeft i,_ => i
  | .reduction i,_ => i
  | _,owner => owner

/-- The compiler distributes eight warps along N for D256, and also for
the 64-row / padded-D128 profile. Other admitted profiles distribute M. -/
def crossWarp (a : Host.Arguments) : Bool :=
  decide (128 < a.q.dims ∨ (64 < a.q.dims ∧ (Host.layout a).BM=64))

def plan (a : Host.Arguments) : Native.Plan where
  route := fun _ edge owner => route edge owner
  reverseMul := fun expression _ => reverse (form expression)
  tree := fun n positive _ owner => reductionTree (crossWarp a) n positive owner


/-- A previous PV output has a rescaled live seed. Even when the current
future-tile guard selects literal one, its multiplication keeps old output
on the left. Only QK's literal zero seed triggers scale reversal. -/
theorem pv_literal_factor_order (a : Host.Arguments) (seed factor : Expr)
    (D row col : Nat) (q k : Fin D → Expr) (literal : ℝ) (owner : Owner) :
    Native.lower (plan a)
      (.binary .mul (.binary .add (.binary .mul seed factor) (.pendingDot D row col q k))
        (.literal literal)) owner=
      .mulRN
        (Native.lower (plan a)
          (.binary .add (.binary .mul seed factor) (.pendingDot D row col q k)) owner)
        (.literal32 literal) := rfl

/-- Arbitrary payload substitution used solely to prove selector independence. -/
def replacePayload (word : BitVec 32 → BitVec 32) (literal : ℝ → ℝ) : Expr → Expr
  | .bits b => .bits (word b)
  | .literal x => .literal (literal x)
  | .binary op a b => .binary op (replacePayload word literal a) (replacePayload word literal b)
  | .exponential x => .exponential (replacePayload word literal x)
  | .materialize x => .materialize (replacePayload word literal x)
  | .reduction op n values => .reduction op n (fun i => replacePayload word literal (values i))
  | .pendingDot d r c q k => .pendingDot d r c
      (fun i => replacePayload word literal (q i)) (fun i => replacePayload word literal (k i))

theorem form_replacePayload (word : BitVec 32 → BitVec 32) (literal : ℝ → ℝ) (e : Expr) :
    form (replacePayload word literal e)=form e := by
  induction e <;> simp_all [replacePayload,form]

theorem selector_payload_independent (a : Host.Arguments)
    (word : BitVec 32 → BitVec 32) (literal : ℝ → ℝ) (e : Expr) (owner : Owner)
    (edge : Edge) (n : Nat) (positive : 0<n) :
    (plan a).route (replacePayload word literal e) edge owner=(plan a).route e edge owner ∧
    (plan a).reverseMul (replacePayload word literal e) owner=(plan a).reverseMul e owner ∧
    (plan a).tree n positive (replacePayload word literal e) owner=(plan a).tree n positive e owner := by
  simp [plan,form_replacePayload]

#axiomsClean selector_payload_independent
#axiomsClean pv_literal_factor_order
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelNative
