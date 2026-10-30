import bench.tritonbench_g.attention_fwd_triton1.Proofs.Transport
import bench.tritonbench_g.attention_fwd_triton1.Proofs.RealOutput
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
noncomputable section

namespace SourceFP
open VeriTile.Triton

/-- Concrete FP32 software scalar profile: round-to-nearest-even, gradual
underflow, signed zero and the explicit canonical-NaN policy in ScalarOps.
Conformance of the actual native opcodes is a separate local obligation. -/
def value (bits : BitVec 32) : FP.Value .fp32 := ⟨bits⟩
def scalar : ISA :=
  ⟨fun a b => (FP.mul .ieeeValue (value a) (value b)).bits,
   fun a b c => (FP.fma .ieeeValue (value a) (value b) (value c)).bits⟩

/-- The scale is the actual FP32 word passed by the host/JIT. No equation to
1/sqrt(D) is required or asserted. Every source rounding site is preserved. -/
def output (c : Shape) (xs : Inputs (BitVec 32)) (actualScale : BitVec 32)
    (head row j : Nat) (ifcond : Bool) : BitVec 32 :=
  Reference.output (wordOps scalar) c xs actualScale head (row/32) ifcond (row%32) j
def history (c : Shape) (xs : Inputs (BitVec 32)) (actualScale : BitVec 32)
    (head chunk i j : Nat) (ifcond : Bool) : BitVec 32 :=
  Reference.carry (wordOps scalar) c xs actualScale head ifcond chunk i j

/-- Exactly two numerical equation families. Both are fixed scalar bit
semantics; neither refers to a dot, output, execution, or Real target. -/
structure LocalConformance (native : ISA) (dom : NumericDomain) : Prop where
  mulRN : ∀ a b, dom.mul a b → native.mulRN a b=scalar.mulRN a b
  fmaRN : ∀ a b c, dom.fma a b c → native.fmaRN a b c=scalar.fmaRN a b c

theorem transport (native : ISA) (dom : NumericDomain) (laws : LocalConformance native dom)
    (x : Expr) (hx : x.Admitted native dom) : x.eval native=x.eval scalar := by
  induction x with
  | word b => rfl
  | zero => rfl
  | cast32 x ih => exact ih hx
  | mulRN x y ihx ihy =>
    rw [Expr.eval,laws.mulRN _ _ hx.2.2,ihx hx.1,ihy hx.2.1]
    rfl
  | fmaRN x y z ihx ihy ihz =>
    rw [Expr.eval,laws.fmaRN _ _ _ hx.2.2.2,ihx hx.1,ihy hx.2.1,ihz hx.2.2.1]
    rfl

def allOperands : NumericDomain := ⟨fun _ _ => True,fun _ _ _ => True⟩
theorem all_admitted (native : ISA) (x : Expr) : x.Admitted native allOperands := by
  induction x <;> simp_all [Expr.Admitted,allOperands]
theorem reference_conforms (dom : NumericDomain) : LocalConformance scalar dom :=
  ⟨fun _ _ _ => rfl,fun _ _ _ _ => rfl⟩
end SourceFP

/-- Every computed output lane and carry update is checked, including masked
tail arithmetic. This is an operand-domain condition, not an answer oracle. -/
structure GraphDomain (native : ISA) (dom : NumericDomain) (c : Shape)
    (xs : Inputs (BitVec 32)) (actualScale : BitVec 32) (ifcond : Bool) : Prop where
  output : ∀ head chunk i j, head<c.heads → chunk<c.chunks → i<32 → j<c.D →
    (Graph.output c xs actualScale head chunk ifcond i j).Admitted native dom
  carry : ∀ head chunk i j, head<c.heads → chunk≤c.chunks → i<c.D → j<c.D →
    (Graph.carry c xs actualScale head ifcond chunk i j).Admitted native dom

structure BitNumerical (native : ISA) (dom : NumericDomain) (c : Shape)
    (xs : Inputs (BitVec 32)) (actualScale : BitVec 32) (ifcond : Bool) : Prop where
  scalarLaws : SourceFP.LocalConformance native dom
  admitted : GraphDomain native dom c xs actualScale ifcond

/-- Closed satisfiability witness for all dimensions, flags, raw inputs and
actual host-scale words. This is not a hardware conformance theorem. -/
theorem reference_witness (c : Shape) (xs : Inputs (BitVec 32)) (actualScale : BitVec 32) (ifcond : Bool) :
    BitNumerical SourceFP.scalar SourceFP.allOperands c xs actualScale ifcond := by
  refine ⟨SourceFP.reference_conforms _,?_,?_⟩ <;> intros <;> apply SourceFP.all_admitted

namespace FPEquiv
theorem output (native : ISA) (dom : NumericDomain) (c : Shape) (abi : ABI c)
    (actualScale : BitVec 32) (store ifcond : Bool) (initial : Memory (BitVec 32))
    (numeric : BitNumerical native dom c (inputsOf c abi initial) actualScale ifcond)
    (head row j : Nat) (hh : head<c.heads) (hr : row<c.T) (hj : j<c.D) :
    Source.run (wordOps native) c abi actualScale store ifcond initial (abi.o.base+c.mainOffset head row j)=
      SourceFP.output c (inputsOf c abi initial) actualScale head row j ifcond := by
  rw [Source.run_output (wordOps native) c abi actualScale store ifcond initial head row j hh hr hj]
  change Reference.output (wordOps native) c (inputsOf c abi initial)
    actualScale head (row/32) ifcond (row%32) j=_
  rw [← Graph.eval_output native c (inputsOf c abi initial) actualScale head (row/32) ifcond (row%32) j]
  rw [SourceFP.transport native dom numeric.scalarLaws _
    (numeric.admitted.output head (row/32) (row%32) j hh (c.chunk_of_row row hr)
      (Nat.mod_lt row (by decide)) hj),Graph.eval_output]
  rfl

theorem history (native : ISA) (dom : NumericDomain) (c : Shape) (abi : ABI c)
    (actualScale : BitVec 32) (store ifcond : Bool) (initial : Memory (BitVec 32))
    (numeric : BitNumerical native dom c (inputsOf c abi initial) actualScale ifcond)
    (hs : store=Bool.true) (head chunk i j : Nat)
    (hh : head<c.heads) (hc : chunk<c.chunks) (hi : i<c.D) (hj : j<c.D) :
    Source.run (wordOps native) c abi actualScale store ifcond initial (abi.h.base+c.historyOffset head chunk i j)=
      SourceFP.history c (inputsOf c abi initial) actualScale head chunk i j ifcond := by
  rw [Source.run_history (wordOps native) c abi actualScale store ifcond initial hs head chunk i j hh hc hi hj]
  change Reference.carry (wordOps native) c (inputsOf c abi initial) actualScale head ifcond chunk i j=_
  rw [← Graph.eval_carry native c (inputsOf c abi initial) actualScale head ifcond chunk i j]
  rw [SourceFP.transport native dom numeric.scalarLaws _
    (numeric.admitted.carry head chunk i j hh (Nat.le_of_lt hc) hi hj),Graph.eval_carry]
  rfl
end FPEquiv

#specNonCircular SourceFP.output avoiding [Source.run,Source.step]
#specNonCircular SourceFP.history avoiding [Source.run,Source.step]
#specNonCircular SourceFP.LocalConformance avoiding [Graph.output,Graph.carry,Source.run,MathTarget.output]
#axiomsClean reference_witness
#axiomsClean FPEquiv.output
#axiomsClean FPEquiv.history
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
