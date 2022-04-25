(define (problem depotprob7512) (:domain Depot)
(:objects
    depot0
    distributor0 distributor1
    truck0 truck1 truck2 truck3
    pallet0 pallet1 pallet2
    crate0 crate1 crate2
    hoist0 hoist1 hoist2
)
(:init
    (pallet pallet0) (surface pallet0) (at pallet0 depot0)
    (pallet pallet1) (surface pallet1) (at pallet1 distributor0)
    (pallet pallet2) (surface pallet2) (at pallet2 distributor1) (clear pallet2)

    (truck truck0) (at truck0 depot0)
    (truck truck1) (at truck1 depot0)
    (truck truck2) (at truck2 depot0)

    (hoist hoist0) (available hoist0) (at hoist0 depot0)
    (hoist hoist1) (available hoist1) (at hoist1 distributor0)
    (hoist hoist2) (available hoist2) (at hoist2 distributor1)

    (crate crate0) (surface crate0) (on crate0 pallet1) (at crate0 distributor1)
    (crate crate1) (surface crate1) (on crate1 pallet0) (at crate1 depot0) (clear crate1)
    (crate crate2) (surface crate2) (on crate2 crate0) (at crate2 depot0) (clear crate2)

    (place depot0)
    (place distributor0)
    (place distributor1)
)

(:goal (and
        (on crate2 crate1)
        (on crate1 crate0)
        (on crate0 pallet0)
    )
))
