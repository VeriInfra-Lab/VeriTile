import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedForwardNumbers

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

/-- Reverse carry is materialized at the end of each iteration, so the next
source addition cannot fuse across the preceding decay multiply. -/
def reverseNode (a : Arithmetic α) (s : Shape) (x : Inputs α) (final : Bool)
    (pid : ProgramId) (r key value : Nat) : Staged.Node α :=
  Staged.tagged (if r=0 then .plain else .reverseCarry)
    (programReverseCarry a s x final pid r key value)

@[simp] theorem reverseNode_value (a : Arithmetic α) (s : Shape) (x : Inputs α) (final : Bool)
    (pid : ProgramId) (r key value : Nat) :
    (reverseNode a s x final pid r key value).value=programReverseCarry a s x final pid r key value := rfl

def reverseUpdateNode (a : Arithmetic α) (s : Shape) (x : Inputs α) (final : Bool)
    (pid : ProgramId) (r key value : Nat) : Staged.Node α :=
  let t := s.T-1-r
  Staged.add a (reverseNode a s x final pid r key value)
    (Staged.multiply a
      (Staged.multiply a (Staged.atom (keyLoad a s x.q pid.head t (pid.key s key)))
        (Staged.tagged .scale (a.scale s.DK)))
      (Staged.atom (valueLoad a s x.do_ pid.head t (pid.value s value))))

theorem reverseUpdateNode_eq (a : Arithmetic α) (s : Shape) (x : Inputs α) (final : Bool)
    (pid : ProgramId) (r key value : Nat) (hr : r<s.T) :
    reverseUpdateNode a s x final pid r key value =
      Staged.tagged .reverseState (programAdjoint a s x final pid (s.T-1-r) key value) := by
  have hkind : (reverseNode a s x final pid r key value).kind ≠ .forwardProduct := by
    simp [reverseNode,Staged.tagged]
    split <;> decide
  simp only [reverseUpdateNode,Staged.reverse_step a _ _ _ _ hkind,reverseNode_value]
  have htime : s.T-1-(s.T-1-r)=r := by omega
  simp [programAdjoint,htime]

theorem reverse_decay_eq (a : Arithmetic α) (s : Shape) (x : Inputs α) (final : Bool)
    (pid : ProgramId) (r key value : Nat) :
    Staged.multiply a (reverseUpdateNode a s x final pid r key value)
      (Staged.tagged .beta (decay a (pid.head%s.H))) =
      reverseNode a s x final pid (r+1) key value := by
  have hkind : (reverseNode a s x final pid r key value).kind ≠ .forwardProduct := by
    simp [reverseNode,Staged.tagged]
    split <;> decide
  simp only [reverseUpdateNode,Staged.reverse_step a _ _ _ _ hkind,
    reverseNode_value,Staged.reverse_decay]
  simp [reverseNode,programReverseCarry,foldN]

theorem querySum_eq (a : Arithmetic α) (s : Shape) (x : Inputs α) (seed : Bool)
    (pid : ProgramId) (t : Nat) (plans : Plans) (hBV : 0<s.BV) (covered : plans.dq.Covers s.BV) :
    (fun i : TileIndex [s.BK] => Staged.multiply a
      (Staged.sumValues a plans (shape := [s.BK,s.BV]) ⟨1,by simp⟩ Bool.false
        (fun j => Staged.multiply a (carryNode a s x seed pid (t+1) j.1.val j.2.1.val)
          (Staged.atom (valueLoad a s x.do_ pid.head t (pid.value s j.2.1.val)))) i)
      (Staged.tagged .scale (a.scale s.DK))) =
      fun i => Staged.tagged .scaledQuery (programQ a s plans x seed pid t i.1.val) := by
  simp only [carryNode,show t+1≠0 by omega,if_false,Staged.query_product]
  rw [Staged.query_reduction a plans s.BK s.BV hBV covered
    (fun k v => programForwardCarry a s x seed pid (t+1) k v)
    (fun v => valueLoad a s x.do_ pid.head t (pid.value s v))]
  rfl

theorem keySum_eq (a : Arithmetic α) (s : Shape) (x : Inputs α) (final : Bool)
    (pid : ProgramId) (r : Nat) (plans : Plans) (hr : r<s.T)
    (hBV : 0<s.BV) (covered : plans.dk.Covers s.BV) :
    Staged.sumValues a plans (shape := [s.BK,s.BV]) ⟨1,by simp⟩ Bool.false
      (fun j => Staged.multiply a (reverseUpdateNode a s x final pid r j.1.val j.2.1.val)
        (Staged.atom (valueLoad a s x.v pid.head (s.T-1-r) (pid.value s j.2.1.val)))) =
      fun i => Staged.atom (programK a s plans x final pid (s.T-1-r) i.1.val) := by
  simp only [reverseUpdateNode_eq a s x final pid r _ _ hr,Staged.reverse_product]
  exact Staged.key_reduction a plans s.BK s.BV hBV covered
    (fun k v => programAdjoint a s x final pid (s.T-1-r) k v)
    (fun v => valueLoad a s x.v pid.head (s.T-1-r) (pid.value s v))

theorem valueSum_eq (a : Arithmetic α) (s : Shape) (x : Inputs α) (final : Bool)
    (pid : ProgramId) (r : Nat) (plans : Plans) (hr : r<s.T)
    (hBK : 0<s.BK) (covered : plans.dv.Covers s.BK) :
    Staged.sumValues a plans (shape := [s.BK,s.BV]) ⟨0,by simp⟩ Bool.false
      (fun j => Staged.multiply a (reverseUpdateNode a s x final pid r j.1.val j.2.1.val)
        (Staged.atom (keyLoad a s x.k pid.head (s.T-1-r) (pid.key s j.1.val)))) =
      fun i => Staged.atom (programV a s plans x final pid (s.T-1-r) i.1.val) := by
  simp only [reverseUpdateNode_eq a s x final pid r _ _ hr,Staged.reverse_product]
  exact Staged.value_reduction a plans s.BK s.BV hBK covered
    (fun k v => programAdjoint a s x final pid (s.T-1-r) k v)
    (fun k => keyLoad a s x.k pid.head (s.T-1-r) (pid.key s k))

#axiomsClean reverseUpdateNode_eq
#axiomsClean reverse_decay_eq
#axiomsClean querySum_eq
#axiomsClean keySum_eq
#axiomsClean valueSum_eq
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
