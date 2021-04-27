; map of the depots:                    
; 0000=1111         
; 0*00 11*1         
;----------         
; 0: depot0 area
; 1: depot1 area
; *: depot access point
; =: transit area

(define (problem storage-18)
(:domain storage-propositional)
(:objects
	depot0-1-1 depot0-1-2 depot0-1-3 depot0-1-4 depot0-2-1 depot0-2-2 depot0-2-3 depot0-2-4 depot1-1-1 depot1-1-2 depot1-1-3 depot1-1-4 depot1-2-1 depot1-2-2 depot1-2-3 depot1-2-4 container-0-0 container-0-1 container-0-2 container-0-3 container-1-0 container-1-1 container-1-2 container-1-3 - storearea
	hoist0 hoist1 hoist2 - hoist
	crate0 crate1 crate2 crate3 crate4 crate5 crate6 crate7 - crate
	container0 container1 - container
	depot0 depot1 - depot
	loadarea transit0 - transitarea)

		(:init
		(at hoist0 depot0-2-2)
		(at hoist1 loadarea)
		(at hoist2 depot1-1-3)
		(available hoist2)
		(clear container-0-3)
		(clear container-1-2)
		(clear container-1-3)
		(clear depot0-1-1)
		(clear depot0-1-2)
		(clear depot0-1-3)
		(clear depot0-1-4)
		(clear depot0-2-1)
		(clear depot0-2-3)
		(clear depot0-2-4)
		(clear depot1-1-1)
		(clear depot1-1-2)
		(clear depot1-1-4)
		(clear depot1-2-1)
		(clear depot1-2-2)
		(clear depot1-2-4)
		(in crate5 depot1)
		(lifting hoist0 crate6)
		(lifting hoist1 crate3)
		(on crate0 container-1-0)
		(on crate1 container-0-1)
		(on crate2 container-0-2)
		(on crate4 container-0-0)
		(on crate5 depot1-2-3)
		(on crate7 container-1-1)
		(connected depot0-1-1 depot0-2-1)
		(connected depot0-1-1 depot0-1-2)
		(connected depot0-1-2 depot0-2-2)
		(connected depot0-1-2 depot0-1-3)
		(connected depot0-1-2 depot0-1-1)
		(connected depot0-1-3 depot0-2-3)
		(connected depot0-1-3 depot0-1-4)
		(connected depot0-1-3 depot0-1-2)
		(connected depot0-1-4 depot0-2-4)
		(connected depot0-1-4 depot0-1-3)
		(connected depot0-2-1 depot0-1-1)
		(connected depot0-2-1 depot0-2-2)
		(connected depot0-2-2 depot0-1-2)
		(connected depot0-2-2 depot0-2-3)
		(connected depot0-2-2 depot0-2-1)
		(connected depot0-2-3 depot0-1-3)
		(connected depot0-2-3 depot0-2-4)
		(connected depot0-2-3 depot0-2-2)
		(connected depot0-2-4 depot0-1-4)
		(connected depot0-2-4 depot0-2-3)
		(connected depot1-1-1 depot1-2-1)
		(connected depot1-1-1 depot1-1-2)
		(connected depot1-1-2 depot1-2-2)
		(connected depot1-1-2 depot1-1-3)
		(connected depot1-1-2 depot1-1-1)
		(connected depot1-1-3 depot1-2-3)
		(connected depot1-1-3 depot1-1-4)
		(connected depot1-1-3 depot1-1-2)
		(connected depot1-1-4 depot1-2-4)
		(connected depot1-1-4 depot1-1-3)
		(connected depot1-2-1 depot1-1-1)
		(connected depot1-2-1 depot1-2-2)
		(connected depot1-2-2 depot1-1-2)
		(connected depot1-2-2 depot1-2-3)
		(connected depot1-2-2 depot1-2-1)
		(connected depot1-2-3 depot1-1-3)
		(connected depot1-2-3 depot1-2-4)
		(connected depot1-2-3 depot1-2-2)
		(connected depot1-2-4 depot1-1-4)
		(connected depot1-2-4 depot1-2-3)
		(connected transit0 depot0-1-4)
		(connected transit0 depot1-1-1)
		(in depot0-1-1 depot0)
		(in depot0-1-2 depot0)
		(in depot0-1-3 depot0)
		(in depot0-1-4 depot0)
		(in depot0-2-1 depot0)
		(in depot0-2-2 depot0)
		(in depot0-2-3 depot0)
		(in depot0-2-4 depot0)
		(in depot1-1-1 depot1)
		(in depot1-1-2 depot1)
		(in depot1-1-3 depot1)
		(in depot1-1-4 depot1)
		(in depot1-2-1 depot1)
		(in depot1-2-2 depot1)
		(in depot1-2-3 depot1)
		(in depot1-2-4 depot1)
		(in crate0 container0)
		(in crate1 container0)
		(in crate2 container0)
		(in crate3 container0)
		(in crate4 container1)
		(in crate5 container1)
		(in crate6 container1)
		(in crate7 container1)
		(in container-0-0 container0)
		(in container-0-1 container0)
		(in container-0-2 container0)
		(in container-0-3 container0)
		(in container-1-0 container1)
		(in container-1-1 container1)
		(in container-1-2 container1)
		(in container-1-3 container1)
		(connected loadarea container-0-0)
		(connected container-0-0 loadarea)
		(connected loadarea container-0-1)
		(connected container-0-1 loadarea)
		(connected loadarea container-0-2)
		(connected container-0-2 loadarea)
		(connected loadarea container-0-3)
		(connected container-0-3 loadarea)
		(connected loadarea container-1-0)
		(connected container-1-0 loadarea)
		(connected loadarea container-1-1)
		(connected container-1-1 loadarea)
		(connected loadarea container-1-2)
		(connected container-1-2 loadarea)
		(connected loadarea container-1-3)
		(connected container-1-3 loadarea)
		(connected depot0-2-2 loadarea)
		(connected loadarea depot0-2-2)
		(connected depot1-2-3 loadarea)
		(connected loadarea depot1-2-3)
		)

(:goal (and
	(in crate0 depot0)
	(in crate1 depot0)
	(in crate2 depot0)
	(in crate3 depot0)
	(in crate4 depot1)
	(in crate5 depot1)
	(in crate6 depot1)
	(in crate7 depot1)))
)

