import bench.repairs.tritonbench_g.parallel_attention.Proofs.Lowering

namespace VeriTile.Bench.Repairs.ParallelAttention.NativePlans
noncomputable section

/-- The retained SM90 artifact pairs. This finite catalog is evidence coverage,
not a restriction silently imposed on the general repaired source API. -/
inductive Profile where
  | tiny | ragged | ordinary | sharded | noscaleRagged | noscaleSharded
  deriving Repr, DecidableEq

def Profile.useScale : Profile → Bool
  | .noscaleRagged | .noscaleSharded => false
  | _ => true

def Profile.isSharded : Profile → Bool
  | .sharded | .noscaleSharded => true
  | _ => false

def Profile.shape : Profile → Shape
  | .tiny => ⟨1,1,1,1,1,16,16,true⟩
  | .ragged => ⟨1,2,33,17,19,32,32,true⟩
  | .ordinary => ⟨2,2,137,48,48,64,64,true⟩
  | .sharded => ⟨1,2,65,127,131,128,128,true⟩
  | .noscaleRagged => ⟨1,2,33,17,19,32,32,false⟩
  | .noscaleSharded => ⟨1,2,65,127,131,128,128,false⟩

/-- A descriptor is extracted from an actual compiled specialization. Shapes
are unrestricted here; the catalog above only supplies retained examples.
All fields are ordinary compiler syntax/ABI data, without numerical claims. -/
structure Descriptor where
  shape : Shape
  postSeed : Bool
  tinyDiagonal : Bool
  scaledSumOnZ : Bool
  paddedCoefficientDirect : Bool

def Profile.descriptor (p : Profile) : Descriptor :=
  ⟨p.shape,p.isSharded,decide (p=.tiny),p.isSharded,decide (p=.tiny)⟩

/-- Same ordered chain as `DotPlan.linear`, with an explicit literal-zero
seed for the sharded DS sites that add dZ after completing the dot. -/
def zeroChain : Nat → DotPlan
  | 0 => .zero
  | n+1 => .fma n (zeroChain n)

theorem zeroChain_leaves (n : Nat) :
    (zeroChain n).leaves=(DotPlan.linear n).leaves := by
  induction n with
  | zero => rfl
  | succ n ih => simp [zeroChain,DotPlan.linear,DotPlan.leaves,ih]

theorem zeroChain_noCarry (n : Nat) : (zeroChain n).carries=0 := by
  induction n with
  | zero => rfl
  | succ n ih => simpa [zeroChain,DotPlan.carries] using ih

def postSeedRight (n : Nat) : DotPlan := .add (zeroChain n) .acc
def postSeedLeft (n : Nat) : DotPlan := .add .acc (zeroChain n)

theorem postSeedRight_complete (n : Nat) : (postSeedRight n).Covers n := by
  constructor
  · simpa [postSeedRight,DotPlan.leaves,zeroChain_leaves] using (DotPlan.linear_complete n).1
  · simp [postSeedRight,DotPlan.carries,zeroChain_noCarry]

theorem postSeedLeft_complete (n : Nat) : (postSeedLeft n).Covers n := by
  constructor
  · simpa [postSeedLeft,DotPlan.leaves,zeroChain_leaves] using (DotPlan.linear_complete n).1
  · simp [postSeedLeft,DotPlan.carries,zeroChain_noCarry]

/-- NativeGraph keys explicitly retain physical value shard and diagonal bit.
For shard zero, sharded DQ adds dot+dZ; sharded DK adds dZ+dot. -/
def dotTree (d : Descriptor) (site : List Nat) (n : Nat) : DotPlan :=
  if d.postSeed && decide (site[4]?.getD 0=0) then
    if site[0]?.getD 9=1 ∧ site[1]?.getD 9=0 then postSeedRight n
    else if site[0]?.getD 9=2 ∧ site[1]?.getD 9=1 ∧
      (site.getLast?.getD 0=1 ∨ d.shape.useScale=false) then postSeedLeft n
    else DotPlan.linear n
  else DotPlan.linear n

theorem dotTree_complete (d : Descriptor) (site : List Nat) (n : Nat) :
    (dotTree d site n).Covers n := by
  unfold dotTree
  split
  · split
    · exact postSeedRight_complete n
    · split
      · exact postSeedLeft_complete n
      · exact DotPlan.linear_complete n
  · exact DotPlan.linear_complete n

/-- Each column-owner lane holds four consecutive columns. In the prefix,
the first addition contracts raw0²+roundedSquare1, then raw2² and raw3².
The diagonal retains all four rounded and masked square leaves. -/
def localNorm (tiny diagonal : Bool) (owner : Nat) : DotPlan :=
  let i := 4*owner
  if diagonal then
    if tiny then .add (.product (i+3)) (.add (.product (i+2)) (.add (.product i) (.product (i+1))))
    else .add (.add (.add (.product i) (.product (i+1))) (.product (i+2))) (.product (i+3))
  else .fma (i+3) (.fma (i+2) (.fma i (.product (i+1))))

/-- Exact XOR 4, XOR 2, XOR 1 butterfly at owner zero. No reassociation or
commutativity law is used to identify this expression. -/
def norm32 (tiny diagonal : Bool) : DotPlan :=
  .add
    (.add (.add (localNorm tiny diagonal 0) (localNorm tiny diagonal 4))
      (.add (localNorm tiny diagonal 2) (localNorm tiny diagonal 6)))
    (.add (.add (localNorm tiny diagonal 1) (localNorm tiny diagonal 5))
      (.add (localNorm tiny diagonal 3) (localNorm tiny diagonal 7)))

theorem norm32_complete (tiny diagonal : Bool) :
    (norm32 tiny diagonal).leaves.Perm (List.range 32) := by
  cases tiny <;> cases diagonal <;> decide

theorem norm32_noCarry (tiny diagonal : Bool) : (norm32 tiny diagonal).carries=0 := by
  cases tiny <;> cases diagonal <;> rfl

/-- Unreached lengths have a covered total definition; every native call is
32 wide. The resolver must reject an artifact with a different width. -/
def normTree (d : Descriptor) (site : List Nat) (n : Nat) : DotPlan :=
  if n=32 then norm32 d.tinyDiagonal (decide (site.getLast?.getD 0=1)) else zeroChain n

theorem normTree_complete (d : Descriptor) (site : List Nat) (n : Nat) :
    (normTree d site n).leaves.Perm (List.range n) := by
  unfold normTree
  split
  · subst n; exact norm32_complete _ _
  · rw [zeroChain_leaves]; exact (DotPlan.linear_complete n).1

theorem normTree_noCarry (d : Descriptor) (site : List Nat) (n : Nat) :
    (normTree d site n).carries=0 := by
  unfold normTree
  split
  · exact norm32_noCarry _ _
  · exact zeroChain_noCarry n

def scalar (d : Descriptor) : ScalarLayout where
  scale := fun _ => .left
  twice := fun _ => .addSelf
  scaleTwice := fun _ => .fmaLeft
  scaledSum := fun _ => if d.scaledSumOnZ then .fmaOnZLeft else .fmaLeft
  coefficientOrder := fun site =>
    if site[1]?.getD 9=10 then
      if d.paddedCoefficientDirect && decide (site.getLast?.getD 0=1 ∧
        d.shape.T ≤ 64*(site[3]?.getD 0)+32*(site[6]?.getD 0)+(site[7]?.getD 0)) then .direct
      else .swapped
    else .direct
  coefficientMask := fun site =>
    if site.getLast?.getD 0=1 then
      if d.shape.useScale then .beforeTwice else .afterProduct
    else if site[1]?.getD 9=10 then .afterTwice else .beforeTwice

/-- Host sum plans are separately supplied and separately bound. The twelve
native compiler artifacts contain no Torch host-reduction implementation. -/
def layout (d : Descriptor) (host : Plans) : CompilerLayout where
  plans := { host with dotTree := dotTree d, dotComplete := dotTree_complete d }
  scalar := scalar d
  normTree := normTree d
  normComplete := normTree_complete d
  normNoCarry := normTree_noCarry d

/-- The parameterized native-artifact relation states exact syntax and ABI
shape. Its descriptor must come from the actual-specialization admission
checker under the documented ordinary compiler contract. Merely constructing
this relation is not evidence that an artifact was checked. -/
structure Binding (d : Descriptor) (host : Plans) (s : Shape) (c : CompilerLayout) : Prop where
  shape : s=d.shape
  compilerLayout : c=layout d host

theorem binding_dot (d : Descriptor) (host : Plans) (s : Shape) (c : CompilerLayout)
    (h : Binding d host s c) (site : List Nat) (n : Nat) :
    c.plans.dotTree site n=dotTree d site n := by rw [h.compilerLayout]; rfl

theorem binding_norm (d : Descriptor) (host : Plans) (s : Shape) (c : CompilerLayout)
    (h : Binding d host s c) (site : List Nat) :
    c.normTree site 32=norm32 d.tinyDiagonal (decide (site.getLast?.getD 0=1)) := by
  rw [h.compilerLayout]; rfl

theorem binding_scalar (d : Descriptor) (host : Plans) (s : Shape) (c : CompilerLayout)
    (h : Binding d host s c) : c.scalar=scalar d := by rw [h.compilerLayout]; rfl

theorem binding_exists (d : Descriptor) (host : Plans) :
    Binding d host d.shape (layout d host) := ⟨rfl,rfl⟩

#axiomsClean zeroChain_leaves
#axiomsClean zeroChain_noCarry
#axiomsClean postSeedRight_complete
#axiomsClean postSeedLeft_complete
#axiomsClean dotTree_complete
#axiomsClean norm32_complete
#axiomsClean norm32_noCarry
#axiomsClean normTree_complete
#axiomsClean normTree_noCarry
#axiomsClean binding_dot
#axiomsClean binding_norm
#axiomsClean binding_scalar
#axiomsClean binding_exists

end
end VeriTile.Bench.Repairs.ParallelAttention.NativePlans
