

(define (problem bw-rand-25)
(:domain blocks)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 )
(:init
	(clear b13)
	(clear b24)
	(clear b25)
	(clear b7)
	(handempty)
	(on b10 b12)
	(on b11 b18)
	(on b12 b8)
	(on b13 b1)
	(on b14 b9)
	(on b15 b10)
	(on b16 b22)
	(on b17 b21)
	(on b18 b16)
	(on b2 b20)
	(on b21 b2)
	(on b23 b15)
	(on b24 b23)
	(on b25 b14)
	(on b3 b6)
	(on b4 b17)
	(on b5 b4)
	(on b6 b11)
	(on b7 b19)
	(on b8 b5)
	(on b9 b3)
	(ontable b1)
	(ontable b19)
	(ontable b20)
	(ontable b22)
)
(:goal
(and
(on b1 b2)
(on b2 b3)
(on b3 b4)
(on b4 b5)
(on b5 b6)
(on b6 b7)
(on b7 b8)
(on b8 b9)
(on b9 b10)
(on b10 b11)
(on b11 b12)
(on b12 b13)
(on b13 b14)
(on b14 b15)
(on b15 b16)
(on b16 b17)
(on b17 b18)
(on b18 b19)
(on b19 b20))
)
)



