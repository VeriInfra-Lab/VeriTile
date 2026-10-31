import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Triton.Float.ExpressionRewrite
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.chunk_gated_attention.Sources.Kernels
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSourceFP
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGatedAttention
open scoped VeriTile.Spec
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
theorem contextual_if_unconditional {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before after : List ComputeStmt) (condition : Op .bool [])
    (lhs rhs : List ComputeStmt)
    (hbody : ∀ t, run (engine M) lhs t = run (engine M) rhs t) :
    Contextual M s before after (.ifThen condition lhs) (.ifThen condition rhs) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step]
    cases hc : FP.Structural.evalOp (engine M) none condition t with
    | none => rfl
    | some value =>
      simp only [bind, Option.bind_some]
      cases hv : value PUnit.unit with
      | «false» => rfl
      | «true» => simp only [↓reduceIte]; rw [hbody t]

variable (dtype out_dtype : FloatDType) (input o : RegionName)
    (head time feature T S BT BS BH base : Nat)
def original : Program where
  kernel := NativeSource.cumOriginal dtype out_dtype input o head time feature T S BT BS BH base
  domain := []
def candidate : Program where
  kernel := NativeSource.cumIndexFirst dtype out_dtype input o head time feature T S BT BS BH base
  domain := []

def activeCode (kernel : ComputeKernel) : List ComputeStmt :=
  match (kernel.surfaceBody[4]? : Option ComputeStmt) with
  | some (.ifThen _ code)=>code
  | _=>[]

def tileBaseOp (BT BS : Nat) (swapped : Bool) : Op .int [] :=
  .add .int .nil
    (.add .int .nil
      (if swapped then .mul .int .nil (.ref .int [] "s_s_h") (.ref .int [] "i_bh")
       else .mul .int .nil (.ref .int [] "i_bh") (.ref .int [] "s_s_h"))
      (.mul .int .nil (.mul .int .nil (.ref .int [] "i_t") (.castNatToInt (.constNat BT)))
        (.ref .int [] "s_s_t")))
    (.mul .int .nil (.mul .int .nil (.ref .int [] "i_s") (.castNatToInt (.constNat BS)))
      (.ref .int [] "s_s_d"))

theorem equivalent :
    original dtype out_dtype input o head time feature T S BT BS BH base
      ≡[([] : Spec.Assumptions FP.GuardedFragment)]
    candidate dtype out_dtype input o head time feature T S BT BS BH base := by
  equiv_decompose
  apply contextual_if_unconditional
  intro t
  change Contextual M t
    ((activeCode (NativeSource.cumOriginal dtype out_dtype input o head time feature T S BT BS BH base)).take 5)
    ((activeCode (NativeSource.cumOriginal dtype out_dtype input o head time feature T S BT BS BH base)).drop 6)
    (.assign .int [] "tile_base" (.alg (tileBaseOp BT BS Bool.false)))
    (.assign .int [] "tile_base" (.alg (tileBaseOp BT BS Bool.true)))
  apply contextual_assign_op
  unfold tileBaseOp
  apply op_add
  · apply op_add
    · intro u hu
      change ((u.regs .int [] "i_bh").bind fun a=>(u.regs .int [] "s_s_h").bind fun b=>
        some (FP.Structural.bop Int.mul .nil a b))=
        ((u.regs .int [] "s_s_h").bind fun b=>(u.regs .int [] "i_bh").bind fun a=>
        some (FP.Structural.bop Int.mul .nil b a))
      cases hi:u.regs .int [] "i_bh" <;> cases hs:u.regs .int [] "s_s_h" <;>
        simp only [Option.bind_some,Option.bind_none]
      congr 1
      funext i
      exact Int.mul_comm _ _
    · exact op_refl _ _ _ _
  · exact op_refl _ _ _ _

#print_fp_assumptions equivalent
#axiomsClean equivalent
end VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSourceFP
