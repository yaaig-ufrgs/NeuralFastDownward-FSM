; map of the depots:         
; 0*
;
; 1*
;  1
;----
; 0: depot0 area
; *: depot access point
; =: transit area

(define (problem storage-7)
(:domain storage-propositional)
(:objects
	depot0-2-1 depot0-2-2

	depot1-1-1 depot1-1-2
	           depot1-2-2

	container-0-0 container-0-1 container-0-2 - storearea
	hoist0 hoist1 - hoist
	crate0 crate1 crate2 - crate
	container0 - container
	depot0 depot1 - depot
	loadarea - transitarea)

(:init
	(at hoist0 depot0-2-1)
	(at hoist1 depot1-1-1)
	(available hoist0)
	(clear container-0-0)
	(clear container-0-1)
	(clear container-0-2)
	(clear depot1-1-2)
	(in crate1 depot0)
	(in crate2 depot1)
	(lifting hoist1 crate0)
	(on crate1 depot0-2-2)
	(on crate2 depot1-2-2)
	(connected depot0-2-1 depot0-2-2)
	(connected depot0-2-2 depot0-2-1)
	(connected depot1-1-1 depot1-1-2)
	(connected depot1-1-2 depot1-2-2)
	(connected depot1-1-2 depot1-1-1)
	(connected depot1-2-2 depot1-1-2)
	(in depot0-2-1 depot0)
	(in depot0-2-2 depot0)
	(in depot1-1-1 depot1)
	(in depot1-1-2 depot1)
	(in depot1-2-2 depot1)
	(in container-0-0 container0)
	(in container-0-1 container0)
	(in container-0-2 container0)
	(connected loadarea container-0-0) 
	(connected container-0-0 loadarea)
	(connected loadarea container-0-1) 
	(connected container-0-1 loadarea)
	(connected loadarea container-0-2) 
	(connected container-0-2 loadarea)
	(connected depot0-2-2 loadarea)
	(connected loadarea depot0-2-2)
	(connected depot1-1-2 loadarea)
	(connected loadarea depot1-1-2)
)

(:goal (and
	(in crate0 depot0)
	(in crate1 depot0)
	(in crate2 depot1)
))
)

