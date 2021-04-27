
(define (problem network4new_all_14_3_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b0 b1 b4 b6 b7 b12 b9 b3 b13 b8 b2 b11 b5 - batch-atom
	a1 a2 a3 a4 - area
	s12 s13 s34 s23 - pipe
	

  )
  (:init

    ;; all pipelines segments are in normal state
    		(normal s12)
		(normal s13)
		(normal s34)
		(normal s23)

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
    	(is-product b10 oca1)
	(is-product b0 oca1)
	(is-product b1 gasoleo)
	(is-product b4 oc1b)
	(is-product b6 gasoleo)
	(is-product b7 oc1b)
	(is-product b12 gasoleo)
	(is-product b9 oc1b)
	(is-product b3 gasoleo)
	(is-product b13 lco)
	(is-product b8 lco)
	(is-product b2 rat-a)
	(is-product b11 gasoleo)
	(is-product b5 rat-a)
	

    ;; batch-atoms initially located in areas
    	(on b0 a4)
	(on b1 a1)
	(on b4 a1)
	(on b6 a3)
	(on b12 a2)
	(on b8 a1)
	

    ;; batch-atoms initially located in pipes
    	(first b5 s12)
	(follow b7 b5)
	(last b7 s12)
	(first b10 s13)
	(follow b9 b10)
	(last b9 s13)
	(first b3 s34)
	(last b3 s34)
	(first b2 s23)
	(follow b13 b2)
	(follow b11 b13)
	(last b11 s23)
	
    ;; unitary pipeline segments
    		(not-unitary s12)
		(not-unitary s13)
		(unitary s34)
		(not-unitary s23)

  )
  (:goal (and
    	(on b1 a2)
	(on b4 a4)
	(on b11 a1)
	
  ))
)
