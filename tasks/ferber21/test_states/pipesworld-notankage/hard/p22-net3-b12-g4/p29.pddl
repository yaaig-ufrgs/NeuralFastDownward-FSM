
(define (problem network3new_all_12_4_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b0 b1 b4 b6 b7 b9 b3 b8 b2 b11 b5 - batch-atom
	a1 a2 a3 a4 - area
	s12 s13 s34 - pipe
	

  )
  (:init
	(first b1 s12)
	(first b5 s13)
	(first b7 s34)
	(follow b0 b5)
	(follow b2 b6)
	(follow b4 b0)
	(follow b6 b1)
	(last b2 s12)
	(last b4 s13)
	(last b7 s34)
	(on b10 a4)
	(on b11 a1)
	(on b3 a3)
	(on b8 a3)
	(on b9 a3)
	(pop-updating s13)
	(push-updating s12)
	;; all pipelines segments are in normal state
	(normal s34)
	;; interfaces restrictions
	(may-interface lco lco)
	(may-interface gasoleo gasoleo)
	(may-interface rat-a rat-a)
	(may-interface oca1 oca1)
	(may-interface oc1b oc1b)
	(may-interface lco gasoleo)
	(may-interface gasoleo lco)
	(may-interface lco oca1)
	(may-interface oca1 lco)
	(may-interface lco oc1b)
	(may-interface oc1b lco)
	(may-interface lco rat-a)
	(may-interface rat-a lco)
	(may-interface gasoleo rat-a)
	(may-interface rat-a gasoleo)
	(may-interface gasoleo oca1)
	(may-interface oca1 gasoleo)
	(may-interface gasoleo oc1b)
	(may-interface oc1b gasoleo)
	(may-interface oca1 oc1b)
	(may-interface oc1b oca1)
	;; network topology definition
	(connect a1 a2 s12)
	(connect a1 a3 s13)
	(connect a3 a4 s34)
	;; batch-atoms products
	(is-product b10 oc1b)
	(is-product b0 lco)
	(is-product b1 gasoleo)
	(is-product b4 rat-a)
	(is-product b6 rat-a)
	(is-product b7 lco)
	(is-product b9 gasoleo)
	(is-product b3 gasoleo)
	(is-product b8 oca1)
	(is-product b2 rat-a)
	(is-product b11 gasoleo)
	(is-product b5 oc1b)
	;; batch-atoms initially located in areas
	;; batch-atoms initially located in pipes
	;; unitary pipeline segments
	(not-unitary s12)
	(not-unitary s13)
	(unitary s34)
)
  (:goal (and
    	(on b6 a2)
	(on b3 a3)
	(on b2 a4)
	(on b11 a1)
	
  ))
)

