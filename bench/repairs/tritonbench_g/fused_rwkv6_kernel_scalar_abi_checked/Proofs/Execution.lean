import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Kernels
import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Reference
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Source

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite FusedRwkv6Kernel
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def parts (c : Shape) : Triton.Source.DynamicRange :=
  (Triton.Source.dynamicRange? (kernel c).surfaceBody "_i").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

theorem source_body (c : Shape) : (kernel c).surfaceBody=
    (parts c).before++[.forRange "_i" 0 c.T 1 (parts c).body]++(parts c).after := by
  rfl

/-- Exact scalar operations after reading the FP32 storage channel, including
the explicit source cast. No arithmetic identity, finite-value assumption or compiler law. -/
def ops (M : Algebra α) : Ops α where
  zero := M.literal (some .fp32) .real 0
  loadCast := M.cast (some .fp32) .real .real
  storeCast := M.cast (some .fp32) .real .real
  add := M.binary (some .fp32) .real .add
  mul := M.binary (some .fp32) .real .mul
  exp := M.unary (some .fp32) .exp

def program (s : State α) : FusedRwkv6Kernel.Program := ⟨s.pids 2,s.pids 1,s.pids 0⟩
def regionName : FusedRwkv6Kernel.Region → RegionName
  | .q => "q" | .k => "k" | .v => "v" | .w => "w" | .u => "u"
  | .h0 => "h0" | .o => "o" | .ht => "ht" | .result => "result" | .outside => "outside"
def memory (s : State α) : FusedRwkv6Kernel.Memory α :=
  fun r b => (s.mem (regionName r) b).read .real

/-- Actual pointer register after n advances. Its clipped dead values are
irrelevant only after the separately proved range guard rejects another step. -/
def pointer (c : Shape) (base pitch n : Nat) : Nat :=
  if c.reverse then base+(c.T-1)*pitch-n*pitch else base+n*pitch

theorem pointer_next (c : Shape) (base pitch n : Nat) :
    pointer c base pitch (n+1)=
      if c.reverse then pointer c base pitch n-pitch else pointer c base pitch n+pitch := by
  cases h : c.reverse <;> simp [pointer,h,Nat.add_mul,Nat.sub_sub,Nat.add_assoc]

theorem pointer_live (c : Shape) (base pitch n : Nat) (live : n<c.T) :
    pointer c base pitch n=base+visit c n*pitch := by
  cases h : c.reverse
  · simp [pointer,visit,h]
  · simp only [pointer,visit,h,if_true]
    have hn : n≤c.T-1 := by omega
    have hm : (c.T-1-n)*pitch=(c.T-1)*pitch-n*pitch := Nat.mul_sub_right_distrib _ _ _
    rw [hm]
    have hmul := Nat.mul_le_mul_right pitch hn
    omega

/-- The signed source value and natural projection agree at every actual
dereference. This includes all padded lanes, not only lanes whose mask is Bool.true. -/
theorem pointer_signed_live (c : Shape) (base pitch n : Nat) (live : n<c.T) :
    (pointer c base pitch n : Int)=
      (base : Int)+(if c.reverse then (c.T:Int)-1-(n:Int) else (n:Int))*(pitch:Int) := by
  rw [pointer_live c base pitch n live]
  cases h : c.reverse
  · simp [visit,h]
  · have ht : 1≤c.T := by omega
    have hn : n≤c.T-1 := by omega
    simp [visit,h,Nat.cast_sub hn,Nat.cast_sub ht]

def keyBase (c : Shape) (s : State α) (j : Fin c.BK) : Nat :=
  s.pids 2*c.keyPitch+s.pids 1*c.BK+j.val
def valueBase (c : Shape) (s : State α) (l : Fin c.BV) : Nat :=
  s.pids 2*c.valuePitch+s.pids 0*c.BV+l.val
def outputBase (c : Shape) (s : State α) (l : Fin c.BV) : Nat :=
  (s.pids 2+s.pids 1*c.B*c.H)*(c.T*c.V)+s.pids 0*c.BV+l.val
def keyMask (c : Shape) (s : State α) (j : Fin c.BK) : Bool :=
  decide (s.pids 1*c.BK+j.val<c.K)
def valueMask (c : Shape) (s : State α) (l : Fin c.BV) : Bool :=
  decide (s.pids 0*c.BV+l.val<c.V)

def seed (M : Algebra α) (c : Shape) (s : State α) : Values α .real [c.BV,c.BK] :=
  fun i => scalarState (ops M) c (inputs c (memory s)) (s.pids 2)
    (s.pids 1*c.BK+i.2.1.val) (s.pids 0*c.BV+i.1.val) 0
def bias (M : Algebra α) (c : Shape) (s : State α) : Values α .real [c.BK] :=
  fun i => (ops M).loadCast (if keyMask c s i.1 then
    (s.mem "u" ((s.pids 2%c.H)*c.K+i.1.val+s.pids 1*c.BK)).read .real else (ops M).zero)

def entry (M : Algebra α) (c : Shape) (s : State α) :
    {t : State α // run (engine M) (parts c).before s=some t} := by
  by_cases hi : c.initial=Bool.true
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,
    Option.map,Option.getD,run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,boundaryZero,Option.bind,engine,Algebra.withDefaultPrecision,
    resolvePrecision,ComputeDType.eraseDType,numeric,integralMod,numericLt,natLt,
    bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,hi]
    rfl

@[simp] theorem entry_mem (M : Algebra α) (c : Shape) (s : State α) :
    (entry M c s).val.mem=s.mem := by
  by_cases hi : c.initial=Bool.true <;> simp [entry,hi]

end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
