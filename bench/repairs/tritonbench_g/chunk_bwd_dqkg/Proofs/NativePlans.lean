import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.NativeRows
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Native
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

inductive Family where | short | tile32 | aligned | tails | multiblock | unaligned
  deriving DecidableEq
inductive Warps where | four | eight
  deriving DecidableEq
def Warps.count : Warps → Nat | .four=>4 | .eight=>8
structure Profile where
  family : Family
  warps : Warps
  deriving DecidableEq

def Profile.width (p : Profile) : Width :=
  match p.family with | .short=>.w16 | .tile32=>.w32 | _=>.w64

/-- Concrete blocked layouts from the retained TTGIR. Slot order expands the
contiguous per-thread coordinates before stepping by the CTA tile extent. -/
structure Layout where
  rows : Nat
  cols : Nat
  sptR : Nat
  sptC : Nat
  tpwR : Nat
  tpwC : Nat
  warpR : Nat
  warpC : Nat
  rowFast : Bool

def Layout.count (l : Layout) : Nat := l.rows*l.cols/(32*l.warpR*l.warpC)
def Layout.index (l : Layout) (thread slot : Nat) : Nat :=
  let lane:=thread%32
  let warp:=thread/32
  let lc:=l.cols/(l.tpwC*l.warpC)
  let lr:=l.rows/(l.tpwR*l.warpR)
  let r:=if l.rowFast then slot%lr else slot/lc
  let c:=if l.rowFast then slot/lr else slot%lc
  let laneR:=if l.rowFast then lane%l.tpwR else lane/l.tpwC
  let laneC:=if l.rowFast then lane/l.tpwR else lane%l.tpwC
  let warpR:=if l.rowFast then warp%l.warpR else warp/l.warpC
  let warpC:=if l.rowFast then warp/l.warpR else warp%l.warpC
  let row:=r%l.sptR+l.sptR*(laneR+l.tpwR*(warpR+l.warpR*(r/l.sptR)))
  let col:=c%l.sptC+l.sptC*(laneC+l.tpwC*(warpC+l.warpC*(c/l.sptC)))
  row*l.cols+col

def historyLayout (p : Profile) : Layout :=
  let w:=p.warps.count
  match p.family with
  | .short=>⟨16,16,1,1,2,16,w,1,Bool.false⟩
  | .tile32=>⟨32,32,1,1,32,1,1,w,Bool.true⟩
  | .aligned | .multiblock=>⟨64,64,1,4,2,16,w,1,Bool.false⟩
  | .tails | .unaligned=>⟨64,64,1,1,32,1,2,w/2,Bool.true⟩

def correctionLayout (p : Profile) : Layout :=
  let w:=p.warps.count
  match p.family with
  | .short=>⟨64,16,1,1,32,1,2,w/2,Bool.true⟩
  | .tile32=>⟨64,32,1,1,1,32,w,1,Bool.false⟩
  | .aligned=>⟨64,64,4,1,16,2,1,w,Bool.true⟩
  | .multiblock=>⟨64,64,1,1,32,1,2,w/2,Bool.true⟩
  | .tails | .unaligned=>⟨64,64,1,1,1,32,w/2,2,Bool.false⟩

def mmaIndex (cols thread slot : Nat) : Nat :=
  (thread/32*16+thread%32/4+slot%4/2*8)*cols+
    (slot/4*8+thread%4*2+slot%2)

def fusedAdd (a b : Graph) : Graph :=
  match a with | .mul32 x y=>.fma32 x y b | _=>.add32 a b

def productLocal (n : Nat) (xs : Nat → Graph) : Graph :=
  if n=1 then xs 0 else
    (List.range (n-2)).foldl (fun acc j=>fusedAdd (xs (j+2)) acc)
      (fusedAdd (xs 0) (xs 1))

def addLocal (appendThird : Bool) (n : Nat) (xs : Nat → Graph) : Graph :=
  if n=1 then xs 0 else
    (List.range (n-2)).foldl (fun acc j=>
      if appendThird && j==0 then .add32 acc (xs (j+2)) else .add32 (xs (j+2)) acc)
      (.add32 (xs 0) (xs 1))

def butterfly (masks : List Nat) (xs : Nat → Graph) : Nat → Graph :=
  masks.foldl (fun values mask=>fun owner=>
    .add32 (values owner) (.shuffle mask (values (owner ^^^ mask)))) xs

def warpMasks (p : Profile) : List Nat :=
  match p.warps with | .four=>[2,1] | .eight=>[4,2,1]

/-- Exact two-level history reduction. With one value per thread the first
butterfly step fuses the local product into the neighbor-product seed. -/
def historySum (p : Profile) (xs : Nat → Graph) : Graph :=
  let l:=historyLayout p
  let partials:=fun thread=>productLocal l.count (fun slot=>xs (l.index thread slot))
  let first:=if l.count=1 then
      fun thread=>fusedAdd (partials thread) (.shuffle 16 (partials (thread ^^^ 16)))
    else butterfly [16] partials
  let warps:=fun w=>butterfly [8,4,2,1] first (w*32)
  butterfly (warpMasks p) warps 0

/-- Four warps reduce directly in the MMA register layout with local FMA.
Eight warps materialize products and convert to the recorded blocked layout
before ordered adds. The final gate/history FMA is applied by staging. -/
def appendThirdCorrection (p : Profile) : Bool :=
  match p.family with | .tile32 | .aligned | .multiblock=>Bool.true | _=>Bool.false

def correctionSum (p : Profile) (xs : Nat → Graph) : Graph :=
  let partials:=match p.warps with
    | .four=>fun thread=>productLocal (p.width.size/2)
        (fun slot=>xs (mmaIndex p.width.size thread slot))
    | .eight=>fun thread=>addLocal (appendThirdCorrection p) (correctionLayout p).count
        (fun slot=>xs ((correctionLayout p).index thread slot))
  let warps:=fun w=>butterfly [16,8,4,2,1] partials (w*32)
  butterfly (warpMasks p) warps 0

/-- Only short-w4 swaps q/dq and k/dk at the row-product scalar ports. -/
def reverseRowOperands (p : Profile) : Bool := p.family==.short && p.warps==.four

#specNonCircular historySum avoiding [VeriTile.Triton.FP.Structural.exec]
#specNonCircular correctionSum avoiding [VeriTile.Triton.FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Native
