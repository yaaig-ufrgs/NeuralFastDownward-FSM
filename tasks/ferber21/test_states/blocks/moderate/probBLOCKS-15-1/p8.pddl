(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear g)
		(handempty)
		(on a k)
		(on b j)
		(on c d)
		(on d a)
		(on e c)
		(on f m)
		(on g h)
		(on h n)
		(on j i)
		(on k b)
		(on l f)
		(on m o)
		(on n l)
		(on o e)
		(ontable i)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)