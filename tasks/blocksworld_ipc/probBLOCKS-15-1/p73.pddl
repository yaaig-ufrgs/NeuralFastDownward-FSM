(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear b)
		(clear d)
		(clear g)
		(clear l)
		(handempty)
		(on a o)
		(on b k)
		(on d a)
		(on e f)
		(on f n)
		(on g h)
		(on h e)
		(on j i)
		(on k j)
		(on n c)
		(on o m)
		(ontable c)
		(ontable i)
		(ontable l)
		(ontable m)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)
