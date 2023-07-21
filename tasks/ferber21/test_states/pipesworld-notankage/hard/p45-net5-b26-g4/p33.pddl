
(define (problem network5new_all_26_4_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b21 b17 b14 b22 b4 b6 b15 b19 b20 b13 b8 b2 b11 b24 b5 b0 b1 b25 b18 b7 b12 b9 b3 b23 b16 - batch-atom
	a1 a2 a3 a4 a5 - area
	s12 s13 s34 s23 s15 - pipe
	

  )
  (:init
	(first b0 s34)
	(first b14 s15)
	(first b2 s12)
	(first b25 s13)
	(first b5 s23)
	(follow b11 b17)
	(follow b12 b19)
	(follow b15 b2)
	(follow b17 b23)
	(follow b18 b6)
	(follow b19 b25)
	(follow b23 b24)
	(follow b24 b14)
	(follow b4 b5)
	(follow b6 b4)
	(last b0 s34)
	(last b11 s15)
	(last b12 s13)
	(last b15 s12)
	(last b18 s23)
	(normal s12)
	(on b1 a4)
	(on b10 a3)
	(on b13 a1)
	(on b16 a1)
	(on b20 a4)
	(on b21 a2)
	(on b22 a4)
	(on b3 a5)
	(on b7 a4)
	(on b8 a2)
	(on b9 a5)
	(push-updating s13)
	(push-updating s15)
	(push-updating s23)
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
	(connect a1 a5 s15)
	;; batch-atoms products
	(is-product b10 rat-a)
	(is-product b21 rat-a)
	(is-product b17 gasoleo)
	(is-product b14 oca1)
	(is-product b22 oc1b)
	(is-product b4 lco)
	(is-product b6 gasoleo)
	(is-product b15 rat-a)
	(is-product b19 gasoleo)
	(is-product b20 lco)
	(is-product b13 gasoleo)
	(is-product b8 lco)
	(is-product b2 gasoleo)
	(is-product b11 rat-a)
	(is-product b24 oca1)
	(is-product b5 rat-a)
	(is-product b0 lco)
	(is-product b1 lco)
	(is-product b25 lco)
	(is-product b18 oca1)
	(is-product b7 rat-a)
	(is-product b12 lco)
	(is-product b9 lco)
	(is-product b3 oca1)
	(is-product b23 oc1b)
	(is-product b16 oc1b)
	;; batch-atoms initially located in areas
	;; batch-atoms initially located in pipes
	;; unitary pipeline segments
	(not-unitary s12)
	(not-unitary s13)
	(unitary s34)
	(not-unitary s23)
	(not-unitary s15)
)
  (:goal (and
    	(on b17 a4)
	(on b20 a2)
	(on b8 a3)
	(on b12 a5)
	
  ))
)
