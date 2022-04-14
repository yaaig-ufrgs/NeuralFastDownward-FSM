(define (problem depotprob7512) (:domain Depot)
(:objects
	depot0 distributor0 distributor1 distributor2 distributor3 distributor4 truck0 truck1 pallet0 pallet1 crate0 crate1 crate2 crate3 hoist0 hoist1 )
(:init
	(pallet pallet0) (surface pallet0)
	(at pallet0 depot0)

	(pallet pallet1) (surface pallet1)
	(at pallet1 distributor1)

	(truck truck0)
	(at truck0 depot0)

	(truck truck1)
	(at truck1 depot0)

	(hoist hoist0) (available hoist0)
	(at hoist0 depot0)

	(hoist hoist1) (available hoist1)
	(at hoist1 distributor0)

	(crate crate0) (surface crate0)
	(on crate0 crate1)
	(at crate0 distributor1)
	(clear crate0)

	(crate crate1) (surface crate1)
	(at crate1 depot0)
	(on crate1 crate3)
	(clear crate1)

	(crate crate2) (surface crate2)
	(on crate2 pallet0)
	(at crate2 depot0)

	(crate crate3) (surface crate3)
	(on crate3 crate2)
	(at crate3 depot0)

	(place depot0)
	(place distributor0)
	(place distributor1)
	(place distributor2)
	(place distributor3)
	(place distributor4)
)

(:goal (and
		(on crate3 crate2)
		(on crate2 crate0)
		(on crate0 crate1)
		(on crate1 pallet0)
	)
))
