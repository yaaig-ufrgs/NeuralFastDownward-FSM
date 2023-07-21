
(define (problem network4new_all_20_7_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b17 b14 b4 b6 b15 b19 b13 b8 b2 b11 b5 b0 b1 b18 b7 b12 b9 b3 b16 - batch-atom
	a1 a2 a3 a4 - area
	s12 s13 s34 s23 - pipe
	

  )
  (:init
	(first b1 s34)
	(first b10 s23)
	(first b16 s12)
	(first b4 s13)
	(follow b0 b10)
	(follow b11 b13)
	(follow b13 b4)
	(follow b15 b18)
	(follow b18 b0)
	(follow b5 b16)
	(last b1 s34)
	(last b11 s13)
	(last b15 s23)
	(last b5 s12)
	(normal s12)
	(on b12 a2)
	(on b14 a1)
	(on b17 a1)
	(on b19 a2)
	(on b2 a1)
	(on b3 a4)
	(on b6 a1)
	(on b7 a1)
	(on b8 a1)
	(on b9 a2)
	(pop-updating s13)
	(pop-updating s23)
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
	(connect a2 a3 s23)
	;; batch-atoms products
	(is-product b10 oc1b)
	(is-product b17 gasoleo)
	(is-product b14 oc1b)
	(is-product b4 rat-a)
	(is-product b6 oc1b)
	(is-product b15 gasoleo)
	(is-product b19 lco)
	(is-product b13 lco)
	(is-product b8 gasoleo)
	(is-product b2 lco)
	(is-product b11 oca1)
	(is-product b5 oc1b)
	(is-product b0 lco)
	(is-product b1 rat-a)
	(is-product b18 lco)
	(is-product b7 rat-a)
	(is-product b12 oc1b)
	(is-product b9 rat-a)
	(is-product b3 rat-a)
	(is-product b16 oca1)
	;; batch-atoms initially located in areas
	;; batch-atoms initially located in pipes
	;; unitary pipeline segments
	(not-unitary s12)
	(not-unitary s13)
	(unitary s34)
	(not-unitary s23)
)
  (:goal (and
    	(on b10 a3)
	(on b17 a4)
	(on b14 a1)
	(on b6 a3)
	(on b13 a1)
	(on b2 a3)
	(on b5 a3)
	
  ))
)
