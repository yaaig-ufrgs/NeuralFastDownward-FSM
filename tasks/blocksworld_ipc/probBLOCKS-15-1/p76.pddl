(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear d)
		(clear e)
		(clear j)
		(handempty)
		(on a i)
		(on b l)
		(on c g)
		(on d k)
		(on e h)
		(on f n)
		(on g m)
		(on i o)
		(on j a)
		(on k b)
		(on m f)
		(on o c)
		(ontable h)
		(ontable l)
		(ontable n)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)
