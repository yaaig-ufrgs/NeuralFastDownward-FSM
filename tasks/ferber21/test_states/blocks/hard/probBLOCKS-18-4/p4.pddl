

(define (problem bw-rand-18)
(:domain blocks)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 )
(:init
	(clear b1)
	(clear b14)
	(clear b16)
	(clear b17)
	(handempty)
	(on b1 b7)
	(on b11 b6)
	(on b13 b11)
	(on b14 b5)
	(on b15 b8)
	(on b16 b18)
	(on b17 b2)
	(on b18 b15)
	(on b2 b4)
	(on b3 b12)
	(on b5 b10)
	(on b6 b3)
	(on b8 b9)
	(on b9 b13)
	(ontable b10)
	(ontable b12)
	(ontable b4)
	(ontable b7)
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
(on b17 b18))
)
)


