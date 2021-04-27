(define (problem depotprob7654) (:domain depot)
(:objects
	depot0 depot1 depot2 distributor0 distributor1 distributor2 truck0 truck1 pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 crate0 crate1 crate2 crate3 crate4 crate5 hoist0 hoist1 hoist2 hoist3 hoist4 hoist5 )
		(:init
		(at crate0 depot2)
		(at crate2 distributor2)
		(at crate3 depot1)
		(at crate4 depot2)
		(at truck0 distributor2)
		(at truck1 depot1)
		(available hoist0)
		(available hoist1)
		(available hoist2)
		(available hoist3)
		(available hoist5)
		(clear crate0)
		(clear crate2)
		(clear crate3)
		(clear pallet0)
		(clear pallet3)
		(clear pallet4)
		(in crate5 truck0)
		(lifting hoist4 crate1)
		(on crate0 crate4)
		(on crate2 pallet5)
		(on crate3 pallet1)
		(on crate4 pallet2)
		(pallet pallet0)
		(surface pallet0)
		(at pallet0 depot0)
		(pallet pallet1)
		(surface pallet1)
		(at pallet1 depot1)
		(pallet pallet2)
		(surface pallet2)
		(at pallet2 depot2)
		(pallet pallet3)
		(surface pallet3)
		(at pallet3 distributor0)
		(pallet pallet4)
		(surface pallet4)
		(at pallet4 distributor1)
		(pallet pallet5)
		(surface pallet5)
		(at pallet5 distributor2)
		(truck truck0)
		(truck truck1)
		(hoist hoist0)
		(at hoist0 depot0)
		(hoist hoist1)
		(at hoist1 depot1)
		(hoist hoist2)
		(at hoist2 depot2)
		(hoist hoist3)
		(at hoist3 distributor0)
		(hoist hoist4)
		(at hoist4 distributor1)
		(hoist hoist5)
		(at hoist5 distributor2)
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
		(place depot0)
		(place depot1)
		(place depot2)
		(place distributor0)
		(place distributor1)
		(place distributor2)
		)

(:goal (and
		(on crate0 crate4)
		(on crate2 pallet3)
		(on crate3 pallet0)
		(on crate4 pallet5)
	)
))

