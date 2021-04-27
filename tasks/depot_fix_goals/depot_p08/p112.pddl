(define (problem depotprob4321) (:domain depot)
(:objects
	depot0 distributor0 distributor1 truck0 truck1 pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 crate0 crate1 crate2 crate3 crate4 crate5 crate6 crate7 crate8 crate9 hoist0 hoist1 hoist2 )
		(:init
		(at crate0 distributor0)
		(at crate1 distributor0)
		(at crate3 distributor1)
		(at crate4 distributor1)
		(at crate5 distributor0)
		(at crate7 distributor1)
		(at crate8 distributor1)
		(at crate9 distributor1)
		(at truck0 distributor0)
		(at truck1 distributor1)
		(available hoist0)
		(available hoist1)
		(available hoist2)
		(clear crate0)
		(clear crate1)
		(clear crate3)
		(clear pallet0)
		(clear pallet2)
		(clear pallet4)
		(in crate2 truck0)
		(in crate6 truck1)
		(on crate0 crate5)
		(on crate1 pallet1)
		(on crate3 crate7)
		(on crate4 pallet3)
		(on crate5 pallet5)
		(on crate7 crate8)
		(on crate8 crate9)
		(on crate9 crate4)
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
		(at pallet5 distributor0)
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
		(place depot0)
		(place distributor0)
		(place distributor1)
		)

(:goal (and
		(on crate0 pallet3)
		(on crate1 crate0)
		(on crate3 crate8)
		(on crate6 pallet2)
		(on crate7 pallet1)
		(on crate8 pallet4)
		(on crate9 pallet0)
	)
))

