import bench.tritonbench_g.fused_recurrent_retention.Proofs.Support.SurfaceSource
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.UnitRanges

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000

structure Layout where
  qHead : Nat
  qTime : Nat
  qChannel : Nat
  vHead : Nat
  vTime : Nat
  vChannel : Nat
  B : Nat
  H : Nat
  T : Nat
  BK : Nat
  BV : Nat
  DK : Nat
  DV : Nat
  doBatch : Nat
  doHead : Nat
  doTime : Nat
  doChannel : Nat
  dfBatch : Nat
  dfHead : Nat
  dfKey : Nat
  dfValue : Nat

structure ForwardRegions where
  q : RegionName
  k : RegionName
  v : RegionName
  o : RegionName
  initial : RegionName
  final : RegionName

structure BackwardRegions where
  q : RegionName
  k : RegionName
  v : RegionName
  do_ : RegionName
  dq : RegionName
  dk : RegionName
  dv : RegionName
  initial : RegionName
  finalGrad : RegionName

namespace Program

def forward (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool) : ComputeKernel :=
  Surface.forward .fp32 .fp32 r.q r.k r.v r.o r.initial r.final
    p.qHead p.qTime p.qChannel p.vHead p.vTime p.vChannel p.B p.H p.T 0
    p.BK p.BV p.DK p.DV hasSeed storeFinal

def backward (p : Layout) (r : BackwardRegions) (hasSeed hasFinal : Bool) : ComputeKernel :=
  Surface.backward .fp32 .fp32 r.q r.k r.v r.do_ r.dq r.dk r.dv r.initial r.finalGrad
    p.doBatch p.doHead p.doTime p.doChannel p.dfBatch p.dfHead p.dfKey p.dfValue
    p.qHead p.qTime p.qChannel p.vHead p.vTime p.vChannel p.B p.H p.T 0
    p.BK p.BV p.DK p.DV hasSeed hasFinal

def forwardParts (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (forward p r hasSeed storeFinal).surfaceBody "_").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

def backwardFirst (p : Layout) (r : BackwardRegions) (hasSeed hasFinal : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (backward p r hasSeed hasFinal).surfaceBody "i").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

/-- Proof-only decomposition of the actual single-argument counted loop. -/
def countedParts? (code : List ComputeStmt) (index : RegName) : Option Source.DynamicRange :=
  match code with
  | [] => none
  | st::rest =>
    match st with
    | .forLoop name stop body =>
      if name=index then some ⟨[],.constNat 0,.constNat stop,.constNat 1,body,rest⟩
      else (countedParts? rest index).map fun part => {part with before:=st::part.before}
    | _ => (countedParts? rest index).map fun part => {part with before:=st::part.before}

def backwardSecond (p : Layout) (r : BackwardRegions) (hasSeed hasFinal : Bool) : Source.DynamicRange :=
  (countedParts? (backwardFirst p r hasSeed hasFinal).after "_").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

theorem forward_split (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool) :
    (forward p r hasSeed storeFinal).surfaceBody =
      (forwardParts p r hasSeed storeFinal).before ++
        [.forRange "_" 0 p.T 1 (forwardParts p r hasSeed storeFinal).body] ++
        (forwardParts p r hasSeed storeFinal).after := by rfl

theorem backward_split (p : Layout) (r : BackwardRegions) (hasSeed hasFinal : Bool) :
    (backward p r hasSeed hasFinal).surfaceBody =
      (backwardFirst p r hasSeed hasFinal).before ++
        [.forRange "i" 0 p.T 1 (backwardFirst p r hasSeed hasFinal).body] ++
        (backwardSecond p r hasSeed hasFinal).before ++
        [.forLoop "_" p.T (backwardSecond p r hasSeed hasFinal).body] := by rfl

#axiomsClean forward_split
#axiomsClean backward_split
end Program
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
