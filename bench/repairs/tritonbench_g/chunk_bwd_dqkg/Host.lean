import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Spec

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg
open VeriTile.Triton Whole
noncomputable section

def canonicalStrides : List Nat → List Nat
  | [] => []
  | _ :: rest => (rest.map (max · 1)).prod :: canonicalStrides rest

structure Tensor where
  region : RegionName
  shape : List Nat
  strides : List Nat
  device : Nat
  dtype : FloatDType
  deriving DecidableEq

def tensor (device : Nat) (region : RegionName) (shape : List Nat) : Tensor :=
  ⟨region,shape,canonicalStrides shape,device,.fp32⟩

def inputDescriptors (c : Config) (device : Nat) : List Tensor :=
  [tensor device "q" [c.B,c.H,c.T,c.K],tensor device "k" [c.B,c.H,c.T,c.K],
   tensor device "v" [c.B,c.H,c.T,c.V],tensor device "h" [c.B,c.H,c.NT*c.K,c.V],
   tensor device "g" [c.B,c.H,c.T],tensor device "do" [c.B,c.H,c.T,c.V],
   tensor device "dh" [c.B,c.H,c.NT*c.K,c.V]]

def outputDescriptors (c : Config) (device : Nat) : Tensor × Tensor × Tensor :=
  (tensor device "dq" [c.B,c.H,c.T,c.K],tensor device "dk" [c.B,c.H,c.T,c.K],
   tensor device "dg" [c.NK,c.B,c.H,c.T])

/-- Allocation may initialize new output memory arbitrarily. Its only frame
premise concerns preexisting memory; no output value or kernel run is assumed. -/
structure Allocation (before : BlockState) where
  state : BlockState
  frame : ∀ (region : RegionName) (off : Nat),region≠"dq" → region≠"dk" → region≠"dg" →
    state.mem region off=before.mem region off

structure Result where
  state : BlockState
  tensors : Tensor × Tensor × Tensor

def invoke (c : Config) (device : Nat) (before : BlockState) (alloc : Allocation before)
    (order : List (CTA c)) : Option Result :=
  (runGrid c order alloc.state).map fun state => ⟨state,outputDescriptors c device⟩

theorem allocation_inputs (before : BlockState) (alloc : Allocation before) :
    inputs alloc.state=inputs before := by
  have hr (r : RegionName) (ha : r≠"dq") (hb : r≠"dk") (hc : r≠"dg") :
      alloc.state.readMem r=before.readMem r := by
    funext off
    simp only [BlockState.readMem,alloc.frame r off ha hb hc]
  simp only [inputs,RealPost.inputs,hr "q" (by decide) (by decide) (by decide),
    hr "k" (by decide) (by decide) (by decide),hr "v" (by decide) (by decide) (by decide),
    hr "h" (by decide) (by decide) (by decide),hr "g" (by decide) (by decide) (by decide),
    hr "do" (by decide) (by decide) (by decide),hr "dh" (by decide) (by decide) (by decide)]

theorem empty_grid (c : Config) (he : c.B=0 ∨ c.H=0 ∨ c.T=0) : allCTAs c=[] := by
  rcases he with hb|hh|ht
  · simp [allCTAs,Config.heads,hb]
  · simp [allCTAs,Config.heads,hh]
  · simp [allCTAs,Config.NT,ht]

theorem empty_return (c : Config) (device : Nat) (before : BlockState) (alloc : Allocation before)
    (he : c.B=0 ∨ c.H=0 ∨ c.T=0) :
    invoke c device before alloc (allCTAs c)=some ⟨alloc.state,outputDescriptors c device⟩ := by
  simp [invoke,empty_grid c he,runGrid]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg
