import VeriTile.Triton.DSL.PackedIntegerPromotion
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PackedIntegerPromotion
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 18000
set_option maxHeartbeats 1500000
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

def signature (dtype : FloatDType) : CallSignature where
  pointers := [("P",.i32)]
  floatingPointers := [("S",dtype),("Y",dtype)]
  scalars := [("lane",.integer)]
  readPorts := ["P","S"]
  writePorts := ["Y"]
def source (dtype : FloatDType) (P S Y : RegionName) (lane : Nat) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [lane := lane] in triton {
    bits = 4
    width = 8
    shift = (lane % width) * bits
    packed = tl.load(P).to(tl.int32)
    nibble = (packed >> shift) & 0xF
    scale = tl.load(S)
    value = nibble * scale
    tl.store(Y, value)
  }
def expected (dtype : FloatDType) (P S Y : RegionName) (lane : Nat) : ComputeKernel := triton {
  tl.ports inputs=[P,S], outputs=[Y]
  tl.region P = tl.int32, S = DTYPE, Y = DTYPE
  bits = $(4)
  width = $(8)
  shift = ($(lane) % width) * bits
  packed = (tl.load(P)).to(tl.int32)
  nibble = (packed >> shift) & 15
  scale = tl.load(S)
  value = (tl.toReal(nibble)).to(DTYPE) * scale
  tl.store(Y, value)
}
/-- The complete program keeps signed extraction, the conversion before the
multiply, the actual tensor's dtype, and the final storage conversion. -/
theorem full_program (dtype : FloatDType) (P S Y : RegionName) (lane : Nat) :
    source dtype P S Y lane=expected dtype P S Y lane := rfl

def aliasSource (dtype : FloatDType) (P S Y : RegionName) (lane : Nat) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [lane := lane] in triton {
    ptr = P + lane
    packed = tl.load(ptr).to(tl.int32)
    nibble = packed & 0xF
    ptr = S + lane
    scale = tl.load(ptr)
    value = nibble * scale
    tl.store(Y, value)
  }
def aliasExpected (dtype : FloatDType) (P S Y : RegionName) (lane : Nat) : ComputeKernel := triton {
  tl.ports inputs=[P,S], outputs=[Y]
  tl.region P = tl.int32, S = DTYPE, Y = DTYPE
  ptr = P + $(lane)
  packed = (tl.load(ptr)).to(tl.int32)
  nibble = packed & 15
  ptr = S + $(lane)
  scale = tl.load(ptr)
  value = (tl.toReal(nibble)).to(DTYPE) * scale
  tl.store(Y, value)
}
/-- Pointer facts are read at the load, including the same alias rebound from
integer storage to floating storage; no future binding types an earlier load. -/
theorem sequential_aliases (dtype : FloatDType) (P S Y : RegionName) (lane : Nat) :
    aliasSource dtype P S Y lane=aliasExpected dtype P S Y lane := rfl

def initial {A : Type} [Inhabited A] : State A :=
  {mem := fun _ _=>.mk .int (-1),regs := fun _ _ _=>none,
   pids := fun _=>0,numPids := fun _=>0,undef := fun dtype _ _=>defaultValue dtype}
theorem signed_extraction {A : Type} [Inhabited A] (M : Algebra A) :
    ∃ t, run M ((source .fp16 "P" "S" "Y" 3).surfaceBody.take 5) initial=some t ∧
      t.regs .int [] "nibble"=some (fun _=>15) := by
  simp [source,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    initial,Option.bind,State.setReg,numeric,natLt,numericLt,integralFloorDiv,integralMod,bop,
    IntegerExpr.sameShape,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]

example (P S Y : RegionName) : True := by
  fail_if_success
    let bad : ComputeKernel := with_triton_call (signature .fp16) call_args [lane := (0 : Nat)] in triton {
      packed = tl.load(P).to(tl.int32)
      scale = tl.load(S)
      value = packed * scale
      tl.store(Y, value)
    }
  trivial
example (P S Y : RegionName) : True := by
  fail_if_success
    let bad : ComputeKernel := with_triton_call (signature .fp16) call_args [lane := (0 : Nat)] in triton {
      packed = tl.load(P).to(tl.int32)
      value = packed & 0x1000000
      scale = tl.load(S)
      result = value * scale
      tl.store(Y, result)
    }
  trivial
example (P S Y : RegionName) : True := by
  fail_if_success
    let bad : ComputeKernel := with_triton_call (signature .fp16) call_args [lane := (0 : Nat)] in triton {
      ptr = S + lane
      packed = tl.load(ptr)
      ptr = P + lane
      value = packed & 0xF
      scale = tl.load(S)
      result = value * scale
      tl.store(Y, result)
    }
  trivial

def narrowSignature : CallSignature where
  pointers := [("P",.i8),("S",.fp16),("Y",.fp16)]
example (P S Y : RegionName) : True := by
  fail_if_success
    let bad : ComputeKernel := with_triton_call narrowSignature call_args [] in triton {
      packed = tl.load(P)
      value = packed & 0xFF
      scale = tl.load(S)
      result = value * scale
      tl.store(Y, result)
    }
  trivial
def differenceSignature (dtype : FloatDType) : CallSignature where
  pointers := [("P",.i32),("Z",.i32)]
  floatingPointers := [("S",dtype),("Y",dtype)]
  readPorts := ["P","Z","S"]
  writePorts := ["Y"]
def differenceSource (dtype : FloatDType) (P Z S Y : RegionName) : ComputeKernel :=
  with_triton_call (differenceSignature dtype) call_args [] in triton {
    left = tl.load(P).to(tl.int32)
    right = tl.load(Z).to(tl.int32)
    delta = (left & 0xF) - (right & 0xF)
    scale = tl.load(S)
    value = delta * scale
    tl.store(Y, value)
  }
def differenceExpected (dtype : FloatDType) (P Z S Y : RegionName) : ComputeKernel := triton {
  tl.ports inputs=[P,Z,S], outputs=[Y]
  tl.region P=tl.int32, Z=tl.int32, S=DTYPE, Y=DTYPE
  left = (tl.load(P)).to(tl.int32)
  right = (tl.load(Z)).to(tl.int32)
  delta = (left & 15) - (right & 15)
  scale = tl.load(S)
  value = (tl.toReal(delta)).to(DTYPE) * scale
  tl.store(Y, value)
}
theorem signed_difference_program (dtype : FloatDType) (P Z S Y : RegionName) :
    differenceSource dtype P Z S Y=differenceExpected dtype P Z S Y := rfl

def constantSignature : CallSignature where
  pointers := []
  scalars := []
  constexpr := ["N","BLOCK"]
def constantSource (N BLOCK : Nat) : ComputeKernel :=
  with_triton_call constantSignature call_args [] in triton {
    extent: tl.constexpr = BLOCK * triton.cdiv(N, BLOCK)
    twice: tl.constexpr = extent * 2
    value = twice
  }
def constantExpected (N BLOCK : Nat) : ComputeKernel := triton {
  extent = $(BLOCK * ((N + BLOCK - 1) / BLOCK))
  twice = $(BLOCK * ((N + BLOCK - 1) / BLOCK) * 2)
  value = twice
}
theorem readonly_constexpr_assignments (N BLOCK : Nat) :
    constantSource N BLOCK=constantExpected N BLOCK := rfl
example (N BLOCK : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel := with_triton_call constantSignature call_args [] in triton {
      value = extent
      extent: tl.constexpr = BLOCK * triton.cdiv(N, BLOCK)
    }
  trivial
example (N BLOCK : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel := with_triton_call constantSignature call_args [] in triton {
      extent: tl.constexpr = BLOCK * triton.cdiv(N, BLOCK)
      extent += 1
    }
  trivial
def runtimeSignature : CallSignature where
  pointers := []
  scalars := [("N",.integer)]
  constexpr := ["BLOCK"]
example (N BLOCK : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel := with_triton_call runtimeSignature call_args [N := N] in triton {
      extent: tl.constexpr = BLOCK * triton.cdiv(N, BLOCK)
    }
  trivial

#axiomsClean full_program
#axiomsClean sequential_aliases
#axiomsClean signed_extraction
#axiomsClean signed_difference_program
#axiomsClean readonly_constexpr_assignments
end
end VeriTile.Tests.PackedIntegerPromotion
