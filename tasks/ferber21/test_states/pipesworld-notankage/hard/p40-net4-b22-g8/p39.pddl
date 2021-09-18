
(define (problem network4new_all_22_8_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b21 b17 b14 b4 b6 b15 b19 b20 b13 b8 b2 b11 b5 b0 b1 b18 b7 b12 b9 b3 b16 - batch-atom
	a1 a2 a3 a4 - area
	s12 s13 s34 s23 - pipe
	

  )
  (:init
	(first b1 s34)
	(first b12 s23)
	(first b16 s13)
	(first b21 s12)
	(follow b0 b20)
	(follow b11 b18)
	(follow b14 b16)
	(follow b18 b12)
	(follow b2 b14)
	(follow b20 b21)
	(last b0 s12)
	(last b1 s34)
	(last b11 s23)
	(last b2 s13)
	(normal s23)
	(on b10 a4)
	(on b13 a3)
	(on b15 a4)
	(on b17 a1)
	(on b19 a1)
	(on b3 a2)
	(on b4 a4)
	(on b5 a4)
	(on b6 a3)
	(on b7 a4)
	(on b8 a4)
	(on b9 a4)
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
	(connect a2 a3 s23)
	;; batch-atoms products
	(is-product b10 oc1b)
	(is-product b21 lco)
	(is-product b17 oca1)
	(is-product b14 gasoleo)
	(is-product b4 gasoleo)
	(is-product b6 lco)
	(is-product b15 oca1)
	(is-product b19 oc1b)
	(is-product b20 gasoleo)
	(is-product b13 oca1)
	(is-product b8 lco)
	(is-product b2 rat-a)
	(is-product b11 oc1b)
	(is-product b5 oc1b)
	(is-product b0 rat-a)
	(is-product b1 oca1)
	(is-product b18 lco)
	(is-product b7 gasoleo)
	(is-product b12 oca1)
	(is-product b9 lco)
	(is-product b3 oca1)
	(is-product b16 oc1b)
	;; batch-atoms initially located in areas
	;; batch-atoms initially located in pipes
	;; unitary pipeline segments
	(not-unitary s12)
	(not-unitary s13)
	(unitary s34)
	(not-unitary s23)
)
  (:goal (and
    	(on b14 a4)
	(on b6 a1)
	(on b15 a2)
	(on b19 a3)
	(on b8 a1)
	(on b18 a4)
	(on b7 a1)
	(on b3 a3)
	
  ))
)

