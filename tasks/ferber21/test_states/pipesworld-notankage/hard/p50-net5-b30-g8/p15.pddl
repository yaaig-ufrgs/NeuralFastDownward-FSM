
(define (problem network5new_all_30_8_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b21 b17 b14 b27 b22 b4 b28 b6 b15 b19 b29 b20 b13 b8 b2 b11 b24 b5 b0 b1 b25 b18 b7 b12 b9 b3 b26 b23 b16 - batch-atom
	a1 a2 a3 a4 a5 - area
	s12 s13 s34 s23 s15 - pipe
	

  )
  (:init
	(first b1 s34)
	(first b15 s13)
	(first b21 s23)
	(first b29 s15)
	(first b9 s12)
	(follow b12 b29)
	(follow b14 b12)
	(follow b2 b9)
	(follow b20 b5)
	(follow b22 b15)
	(follow b25 b3)
	(follow b26 b2)
	(follow b3 b21)
	(follow b5 b14)
	(follow b6 b22)
	(last b1 s34)
	(last b20 s15)
	(last b25 s23)
	(last b26 s12)
	(last b6 s13)
	(normal s23)
	(on b0 a5)
	(on b10 a2)
	(on b11 a2)
	(on b13 a1)
	(on b16 a4)
	(on b17 a2)
	(on b18 a4)
	(on b19 a2)
	(on b23 a5)
	(on b24 a5)
	(on b27 a2)
	(on b28 a5)
	(on b4 a5)
	(on b7 a2)
	(on b8 a3)
	(push-updating s12)
	(push-updating s13)
	(push-updating s15)
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
	(is-product b10 oca1)
	(is-product b21 oca1)
	(is-product b17 rat-a)
	(is-product b14 gasoleo)
	(is-product b27 oc1b)
	(is-product b22 oc1b)
	(is-product b4 gasoleo)
	(is-product b28 lco)
	(is-product b6 gasoleo)
	(is-product b15 oc1b)
	(is-product b19 gasoleo)
	(is-product b29 oca1)
	(is-product b20 rat-a)
	(is-product b13 rat-a)
	(is-product b8 oca1)
	(is-product b2 rat-a)
	(is-product b11 lco)
	(is-product b24 lco)
	(is-product b5 lco)
	(is-product b0 rat-a)
	(is-product b1 gasoleo)
	(is-product b25 oca1)
	(is-product b18 gasoleo)
	(is-product b7 lco)
	(is-product b12 oca1)
	(is-product b9 rat-a)
	(is-product b3 gasoleo)
	(is-product b26 rat-a)
	(is-product b23 lco)
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
    	(on b17 a3)
	(on b27 a4)
	(on b6 a4)
	(on b20 a2)
	(on b11 a1)
	(on b24 a4)
	(on b18 a3)
	(on b16 a1)
	
  ))
)

