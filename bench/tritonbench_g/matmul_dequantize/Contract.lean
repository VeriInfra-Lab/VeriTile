import bench.tritonbench_g.matmul_dequantize.Proofs.ABI
import VeriTile.Torch.FP32
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Checked
open VeriTile.Triton Host

instance (t : Host.Tensor) (r c : Nat) (d : StorageDType) : Decidable (t.Valid r c d) :=
  inferInstanceAs (Decidable (_ ∧ _))

/-- Exact source-expression guard, including inactive lanes and final pointer
updates. It contains no numerical payload, execution or output assertion. -/
instance boundedDecision : (e : ABI.Expr) → Decidable e.Bounded
  | .value x => inferInstanceAs (Decidable (0≤x ∧ x<2^31))
  | .add a b | .sub a b | .mul a b | .min a b | .div a b | .mod a b => by
      unfold ABI.Expr.Bounded
      letI := boundedDecision a
      letI := boundedDecision b
      unfold ABI.Fits32
      infer_instance

def metadataChecks (a : Arguments) (route : Route) : Bool := decide (
  (0<a.M ∧ 0<a.N ∧ 0<a.K ∧ 0<a.group) ∧
  (a.K%8=0 ∧ a.N%8=0 ∧ a.K%a.group=0) ∧
  a.a.Valid a.M a.K (if route=.s2 then .fp32 else .fp16) ∧
  a.q.Valid (a.K/8) a.N .int32 ∧
  a.s.Valid (a.K/a.group) a.N (if route=.s2 ∨ route=.dequant32 then .fp32 else .fp16) ∧
  a.z.Valid (a.K/a.group) (a.N/8) .int32 ∧ a.c.Valid a.M a.N .fp16 ∧
  a.d.Valid a.K a.N (if route=.dequant32 then .fp32 else .fp16) ∧
  (tensors a).Pairwise (fun x y => x.region≠y.region) ∧
  (∀ t∈tensors a,t.device=a.a.device) ∧
  (∀ t∈tensors a,t.base+bytes t*t.allocation<2^64) ∧
  (route=.gptq → a.K%32=0 ∧ a.group%32=0 ∧ a.N%256=0))

def expressionChecks (a : Arguments) (route : Route) (cfg : FirstConfig) : Bool :=
  (List.range (ABI.count a route cfg)).all fun pid =>
    (ABI.expressions a route cfg pid).all fun e => decide e.Bounded

def allocationChecks (a : Arguments) (route : Route) (cfg : FirstConfig) : Bool :=
  (List.range (ABI.count a route cfg)).all fun pid =>
    (ABI.accesses a route cfg pid).all fun x =>
      !x.active || decide (x.tensor.storageOffset+x.offset<x.tensor.allocation)

def pointerChecks (a : Arguments) : Bool := decide (
  ∀ t∈tensors a,t.base+bytes t*(t.storageOffset+2^31)<2^64)

def candidates (route : Route) : List FirstConfig :=
  if route=.gptq then firstCandidates else [.m64n256]

def indexChecks (a : Arguments) (route : Route) : Bool := pointerChecks a &&
  (candidates route).all fun cfg => expressionChecks a route cfg && allocationChecks a route cfg

inductive Error where | dynamic | metadata | indices deriving DecidableEq, Repr

def checkInputs (a : Arguments) (route : Route) : Except Error Unit :=
  if !metadataChecks a route then .error .metadata
  else if !indexChecks a route then .error .indices else .ok ()
def InputContract (a : Arguments) (route : Route) :=
  metadataChecks a route=Bool.true ∧ indexChecks a route=Bool.true

theorem checkInputs_iff (a : Arguments) (route : Route) :
    checkInputs a route=.ok () ↔ InputContract a route := by simp [checkInputs,InputContract,ite_eq_iff]

theorem admitted (a : Arguments) (route : Route) (h : InputContract a route) : Admissible a route := by
  have h := h.1
  simp only [metadataChecks,decide_eq_true_eq] at h
  obtain ⟨positive,packed,av,qv,sv,zv,cv,dv,separate,devices,pointers,gptq⟩ := h
  exact ⟨positive,packed,av,qv,sv,zv,cv,dv,⟨qv.2.2.1,zv.2.2.1⟩,separate,devices,pointers,
    fun hr => ⟨(gptq hr).1,(gptq hr).2.1⟩,fun hr => (gptq hr).2.2⟩

theorem indices (a : Arguments) (route : Route) (cfg : FirstConfig) (hc : cfg∈candidates route)
    (h : InputContract a route) : ABI.IndexABI a route cfg := by
  have hi := h.2
  simp only [indexChecks,Bool.and_eq_true,List.all_eq_true] at hi
  obtain ⟨expr,alloc⟩ := hi.2 cfg hc
  simp only [expressionChecks,List.all_eq_true,List.mem_range,decide_eq_true_eq] at expr
  simp only [allocationChecks,List.all_eq_true,List.mem_range,Bool.or_eq_true,Bool.not_eq_true',decide_eq_true_eq] at alloc
  refine ⟨expr,?_,?_⟩
  · intro pid hp x hx active
    rcases alloc pid hp x hx with inactive|bound
    · simp_all
    · exact bound
  · intro t ht x hx
    have hp := hi.1
    simp only [pointerChecks,decide_eq_true_eq] at hp
    have room := hp t ht
    constructor
    · have nonnegative := hx.1
      positivity
    · have bound : (t.base : Int)+bytes t*((t.storageOffset : Int)+x)≤
          t.base+bytes t*(t.storageOffset+2^31) := by
        have high := hx.2
        have bw := Nat.zero_le (bytes t)
        push_cast
        nlinarith
      exact lt_of_le_of_lt bound (by exact_mod_cast room)

/-- Unsupported Python type/rank/layout/dtype/device/lazy-view arguments are
classified before the ordinary descriptor and storage bindings are queried. -/
inductive PublicArguments where
  | unsupported
  | supported (arguments : Arguments) (route : Route)
def checkArguments : PublicArguments → Except Error Unit
  | .unsupported => .error .dynamic
  | .supported a route => checkInputs a route
def AdmittedArguments : PublicArguments → Prop
  | .unsupported => False
  | .supported a route => InputContract a route
theorem checkArguments_iff (p : PublicArguments) : checkArguments p=.ok () ↔ AdmittedArguments p := by
  cases p <;> simp [checkArguments,AdmittedArguments,checkInputs_iff]
#axiomsClean checkArguments_iff

#specNonCircular InputContract avoiding [FirstGrid.program,SecondGrid.program,DequantGrid16.program,DequantGrid32.program]
#axiomsClean checkInputs_iff
#axiomsClean admitted
#axiomsClean indices
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Checked
