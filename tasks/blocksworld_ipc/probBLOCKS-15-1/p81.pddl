(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear c)
		(clear e)
		(clear h)
		(handempty)
		(on a k)
		(on b j)
		(on c n)
		(on d a)
		(on e d)
		(on f o)
		(on h m)
		(on j i)
		(on k b)
		(on l g)
		(on m f)
		(on n l)
		(ontable g)
		(ontable i)
		(ontable o)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)
