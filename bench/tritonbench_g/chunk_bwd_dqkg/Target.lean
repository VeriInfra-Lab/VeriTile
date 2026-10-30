import bench.tritonbench_g.chunk_bwd_dqkg.IO
import bench.tritonbench_g.chunk_bwd_dqkg.Spec

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Target
open VeriTile Triton FP.Structural Whole KernelNativeCarrier
noncomputable section

def qCoordinates (c : Config) (n : Nat) (h : n<c.heads*c.T*c.K) : Fin c.heads × Fin c.T × Fin c.K :=
  let i : Fin (c.heads*c.T*c.K):=⟨n,h⟩
  (i.divNat.divNat,i.divNat.modNat,i.modNat)
def gCoordinates (c : Config) (n : Nat) (h : n<c.NK*c.heads*c.T) : Fin c.NK × Fin c.heads × Fin c.T :=
  let i : Fin (c.NK*c.heads*c.T):=⟨n,h⟩
  (i.divNat.divNat,i.divNat.modNat,i.modNat)

/-- Independent finite mathematical dQ, dK and the unreduced per-key-tile dG. -/
def mathematical (c : Config) (vals : IO.Inputs ℝ c) : IO.Outputs ℝ c := by
  intro o j
  by_cases h0 : o=0
  · subst o
    let p:=qCoordinates c j.val (by simpa [IO.outputSize,IO.outputRegion,extent] using j.isLt)
    exact Spec.dq c (Spec.views c (inputs c vals)) p.1.val (p.2.1.val/64) (p.2.1.val%64) p.2.2.val
  by_cases h1 : o=1
  · subst o
    let p:=qCoordinates c j.val (by simpa [IO.outputSize,IO.outputRegion,extent] using j.isLt)
    exact Spec.dk c (Spec.views c (inputs c vals)) p.1.val (p.2.1.val/64) (p.2.1.val%64) p.2.2.val
  have h2 : o=2 := by omega
  subst o
  let p:=gCoordinates c j.val (by simpa [IO.outputSize,IO.outputRegion,extent] using j.isLt)
  exact Spec.dg c (Spec.views c (inputs c vals)) p.1.val p.2.1.val (p.2.2.val/64) (p.2.2.val%64)

namespace Native
/-- The retained native graph on logical arrays; no interpreter state appears
in these definitions. Materialization, K8 dots, reductions and scale bits are
the same existing instruction model, with no floating-to-Real identity. -/
def gate (isa : Instructions) (route : Routing) (c : Config) (μ : RegionName → Nat → Node)
    (a : CTA c) (r : Fin 64) : Node :=
  materialize isa route (if a.time.val*64+r.val<c.T then
    μ "g" (a.head.val*c.T+a.time.val*64+r.val) else .zero)
def lastGate (isa : Instructions) (route : Routing) (c : Config) (μ : RegionName → Nat → Node)
    (a : CTA c) : Node :=
  materialize isa route (μ "g" (a.head.val*c.T+(min (a.time.val*64+64) c.T-1)))
def value (c : Config) (μ : RegionName → Nat → Node) (a : CTA c) (r : RegionName)
    (block : Nat) (i : TileIndex [64,c.BV]) : Node :=
  if a.time.val*64+i.1.val<c.T ∧ block*c.BV+i.2.1.val<c.V then
    μ r (a.head.val*(c.T*c.V)+(a.time.val*64+i.1.val)*c.V+(block*c.BV+i.2.1.val)) else .zero
def history (isa : Instructions) (route : Routing) (c : Config) (μ : RegionName → Nat → Node)
    (a : CTA c) (r : RegionName) (block : Nat) (i : TileIndex [c.BV,c.BK]) : Node :=
  materialize isa route (if block*c.BV+i.1.val<c.V ∧ a.key.val*c.BK+i.2.1.val<c.K then
    μ r (a.head.val*(c.NT*c.K*c.V)+a.time.val*(c.K*c.V)+
      (block*c.BV+i.1.val)+(a.key.val*c.BK+i.2.1.val)*c.V) else .zero)
def queryKey (c : Config) (μ : RegionName → Nat → Node) (a : CTA c)
    (r : RegionName) (i : TileIndex [64,c.BK]) : Node :=
  if a.time.val*64+i.1.val<c.T ∧ a.key.val*c.BK+i.2.1.val<c.K then
    μ r (a.head.val*(c.T*c.K)+(a.time.val*64+i.1.val)*c.K+(a.key.val*c.BK+i.2.1.val)) else .zero
abbrev Carry := KernelNativeReference.Carry
abbrev seed := KernelNativeReference.seed
abbrev total := @KernelNativeReference.total
def step (isa : Instructions) (route : Routing) (c : Config) (μ : RegionName → Nat → Node)
    (a : CTA c) (block : Nat) (carry : Carry c) : Carry c :=
  let v:=value c μ a "v" block
  let d:=value c μ a "do" block
  let h:=history isa route c μ a "h" block
  let f:=history isa route c μ a "dh" block
  ⟨fun i=>.add (carry.q i) (.dot c.BV i.1.val i.2.1.val (fun j=>d (i.1,j,())) (fun j=>h (j,i.2.1,()))),
   fun i=>.add (carry.k i) (.dot c.BV i.1.val i.2.1.val (fun j=>v (i.1,j,())) (fun j=>f (j,i.2.1,()))),
   fun i=>.add (carry.score i) (.dot c.BV i.1.val i.2.1.val (fun j=>d (i.1,j,())) (fun j=>v (i.2.1,j,()))),
   .add carry.terminal (total route h f)⟩
def accumulate (isa : Instructions) (route : Routing) (c : Config) (μ : RegionName → Nat → Node)
    (a : CTA c) : Nat → Carry c
  | 0=>seed c
  | n+1=>step isa route c μ a n (accumulate isa route c μ a n)
def gatedQ (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (μ : RegionName → Nat → Node) (a : CTA c) (carry : Carry c) (i : TileIndex [64,c.BK]) : Node :=
  .mul (.mul (carry.q i) (.exp (gate isa route c μ a i.1))) (.scale scale)
def gatedK (isa : Instructions) (route : Routing) (c : Config)
    (μ : RegionName → Nat → Node) (a : CTA c) (carry : Carry c) (i : TileIndex [64,c.BK]) : Node :=
  .mul (carry.k i) (.exp (.add (.neg (gate isa route c μ a i.1)) (lastGate isa route c μ a)))
def scores (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (μ : RegionName → Nat → Node) (a : CTA c) (carry : Carry c) (i : TileIndex [64,64]) : Node :=
  if i.2.1.val ≤ i.1.val then .mul (.mul (carry.score i) (.scale scale))
    (.exp (.sub (gate isa route c μ a i.1) (gate isa route c μ a i.2.1))) else .zero
def query (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (μ : RegionName → Nat → Node) (a : CTA c) (carry : Carry c) (i : TileIndex [64,c.BK]) : Node :=
  .add (gatedQ isa route scale c μ a carry i) (.dot 64 i.1.val i.2.1.val
    (fun j=>scores isa route scale c μ a carry (i.1,j,())) (fun j=>queryKey c μ a "k" (j,i.2.1,())))
def key (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (μ : RegionName → Nat → Node) (a : CTA c) (carry : Carry c) (i : TileIndex [64,c.BK]) : Node :=
  .add (gatedK isa route c μ a carry i) (.dot 64 i.1.val i.2.1.val
    (fun j=>scores isa route scale c μ a carry (j,i.1,())) (fun j=>queryKey c μ a "q" (j,i.2.1,())))
def correction (isa : Instructions) (route : Routing) (c : Config)
    (μ : RegionName → Nat → Node) (a : CTA c) (carry : Carry c) : Node :=
  .add (.mul carry.terminal (.exp (lastGate isa route c μ a)))
    (total route (gatedK isa route c μ a carry) (queryKey c μ a "k"))
def gateResult (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (μ : RegionName → Nat → Node) (a : CTA c) (carry : Carry c) (i : TileIndex [64]) : Node :=
  let row:=Node.add .zero (reduce route (shape:=[64,c.BK]) ⟨1,by simp⟩ Bool.false
    (fun j=>.sub (.mul (queryKey c μ a "q" j) (query isa route scale c μ a carry j))
      (.mul (queryKey c μ a "k" j) (key isa route scale c μ a carry j))) i)
  if i.1.val<min 64 (c.T-a.time.val*64)-1 then row else .add row (correction isa route c μ a carry)
end Native

def native (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (vals : IO.Inputs Node c) : IO.Outputs Node c := by
  let μ:=inputs c vals
  intro o j
  by_cases h0 : o=0
  · subst o
    let p:=qCoordinates c j.val (by simpa [IO.outputSize,IO.outputRegion,extent] using j.isLt)
    let a:=qOwner c p.1 p.2.1 p.2.2
    exact Native.query isa route scale c μ a (Native.accumulate isa route c μ a (KernelNative.blocks c))
      (rowLane p.2.1.val,keyLane c p.2.2.val,())
  by_cases h1 : o=1
  · subst o
    let p:=qCoordinates c j.val (by simpa [IO.outputSize,IO.outputRegion,extent] using j.isLt)
    let a:=qOwner c p.1 p.2.1 p.2.2
    exact Native.key isa route scale c μ a (Native.accumulate isa route c μ a (KernelNative.blocks c))
      (rowLane p.2.1.val,keyLane c p.2.2.val,())
  have h2 : o=2 := by omega
  subst o
  let p:=gCoordinates c j.val (by simpa [IO.outputSize,IO.outputRegion,extent] using j.isLt)
  let a:=gOwner c p.1 p.2.1 p.2.2
  exact Native.gateResult isa route scale c μ a (Native.accumulate isa route c μ a (KernelNative.blocks c))
    (rowLane p.2.2.val,())

#specNonCircular mathematical avoiding [Whole.runGrid,VeriTile.Triton.exec]
#specNonCircular native avoiding [KernelNative.runQuoted,Typed.runGrid,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Target
