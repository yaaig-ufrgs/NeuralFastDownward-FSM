
(define (problem network3new_all_12_2_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b0 b1 b4 b6 b7 b9 b3 b8 b2 b11 b5 - batch-atom
	a1 a2 a3 a4 - area
	s12 s13 s34 - pipe
	

  )
  		(:init
		(first b2 s12)
		(first b4 s34)
		(first b5 s13)
		(follow b1 b2)
		(follow b11 b5)
		(follow b7 b1)
		(follow b9 b11)
		(last b4 s34)
		(last b7 s12)
		(last b9 s13)
		(on b0 a2)
		(on b10 a4)
		(on b3 a4)
		(on b6 a4)
		(on b8 a3)
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
	(on b11 a1)
	
  ))
)

