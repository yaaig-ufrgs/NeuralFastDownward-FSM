(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear a)
		(clear d)
		(clear g)
		(clear o)
		(handempty)
		(on a l)
		(on b j)
		(on c h)
		(on d c)
		(on e n)
		(on g k)
		(on h e)
		(on j i)
		(on k b)
		(on m f)
		(on n m)
		(ontable f)
		(ontable i)
		(ontable l)
		(ontable o)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)
