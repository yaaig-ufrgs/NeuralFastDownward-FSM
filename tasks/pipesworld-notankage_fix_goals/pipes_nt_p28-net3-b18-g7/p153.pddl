
(define (problem network3new_all_18_7_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b17 b14 b4 b6 b15 b13 b8 b2 b11 b5 b0 b1 b7 b12 b9 b3 b16 - batch-atom
	a1 a2 a3 a4 - area
	s12 s13 s34 - pipe
	

  )
  		(:init
		(first b1 s13)
		(first b15 s12)
		(first b16 s34)
		(follow b10 b3)
		(follow b11 b1)
		(follow b3 b15)
		(last b10 s12)
		(last b11 s13)
		(last b16 s34)
		(normal s13)
		(on b0 a1)
		(on b12 a1)
		(on b13 a2)
		(on b14 a2)
		(on b17 a4)
		(on b2 a4)
		(on b4 a4)
		(on b5 a3)
		(on b6 a4)
		(on b7 a3)
		(on b8 a3)
		(on b9 a1)
		(pop-updating s12)
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
		(is-product b10 gasoleo)
		(is-product b17 oca1)
		(is-product b14 gasoleo)
		(is-product b4 oc1b)
		(is-product b6 gasoleo)
		(is-product b15 lco)
		(is-product b13 oca1)
		(is-product b8 oca1)
		(is-product b2 gasoleo)
		(is-product b11 oca1)
		(is-product b5 lco)
		(is-product b0 oc1b)
		(is-product b1 oc1b)
		(is-product b7 oca1)
		(is-product b12 oca1)
		(is-product b9 rat-a)
		(is-product b3 oc1b)
		(is-product b16 rat-a)
		;; batch-atoms initially located in areas
		;; batch-atoms initially located in pipes
		;; unitary pipeline segments
		(not-unitary s12)
		(not-unitary s13)
		(unitary s34)
		)
  (:goal (and
    	(on b17 a1)
	(on b4 a2)
	(on b6 a2)
	(on b11 a3)
	(on b1 a2)
	(on b9 a4)
	(on b3 a3)
	
  ))
)

