(define (problem depotprob5451) (:domain depot)
(:objects
	depot0 distributor0 distributor1 truck0 truck1 pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 crate0 crate1 crate2 crate3 crate4 crate5 crate6 crate7 crate8 crate9 crate10 crate11 crate12 crate13 crate14 hoist0 hoist1 hoist2 )
		(:init
		(at crate0 distributor1)
		(at crate1 depot0)
		(at crate10 distributor1)
		(at crate11 distributor0)
		(at crate12 distributor0)
		(at crate13 distributor0)
		(at crate14 distributor0)
		(at crate2 distributor1)
		(at crate4 distributor1)
		(at crate5 distributor1)
		(at crate7 distributor1)
		(at crate8 distributor0)
		(at truck0 distributor0)
		(at truck1 distributor0)
		(available hoist0)
		(available hoist1)
		(clear crate1)
		(clear crate13)
		(clear crate2)
		(clear crate5)
		(clear crate8)
		(clear pallet5)
		(in crate3 truck1)
		(in crate9 truck0)
		(lifting hoist2 crate6)
		(on crate0 pallet2)
		(on crate1 pallet0)
		(on crate10 crate7)
		(on crate11 pallet4)
		(on crate12 crate11)
		(on crate13 pallet1)
		(on crate14 crate12)
		(on crate2 crate10)
		(on crate4 crate0)
		(on crate5 pallet3)
		(on crate7 crate4)
		(on crate8 crate14)
		(pallet pallet0)
		(surface pallet0)
		(at pallet0 depot0)
		(pallet pallet1)
		(surface pallet1)
		(at pallet1 distributor0)
		(pallet pallet2)
		(surface pallet2)
		(at pallet2 distributor1)
		(pallet pallet3)
		(surface pallet3)
		(at pallet3 distributor1)
		(pallet pallet4)
		(surface pallet4)
		(at pallet4 distributor0)
		(pallet pallet5)
		(surface pallet5)
		(at pallet5 depot0)
		(truck truck0)
		(truck truck1)
		(hoist hoist0)
		(at hoist0 depot0)
		(hoist hoist1)
		(at hoist1 distributor0)
		(hoist hoist2)
		(at hoist2 distributor1)
		(crate crate0)
		(surface crate0)
		(crate crate1)
		(surface crate1)
		(crate crate2)
		(surface crate2)
		(crate crate3)
		(surface crate3)
		(crate crate4)
		(surface crate4)
		(crate crate5)
		(surface crate5)
		(crate crate6)
		(surface crate6)
		(crate crate7)
		(surface crate7)
		(crate crate8)
		(surface crate8)
		(crate crate9)
		(surface crate9)
		(crate crate10)
		(surface crate10)
		(crate crate11)
		(surface crate11)
		(crate crate12)
		(surface crate12)
		(crate crate13)
		(surface crate13)
		(crate crate14)
		(surface crate14)
		(place depot0)
		(place distributor0)
		(place distributor1)
		)

(:goal (and
		(on crate0 crate5)
		(on crate1 crate2)
		(on crate2 crate10)
		(on crate3 pallet0)
		(on crate4 crate6)
		(on crate5 pallet5)
		(on crate6 pallet4)
		(on crate9 crate1)
		(on crate10 pallet2)
		(on crate11 pallet1)
		(on crate12 crate14)
		(on crate13 crate3)
		(on crate14 pallet3)
	)
))

