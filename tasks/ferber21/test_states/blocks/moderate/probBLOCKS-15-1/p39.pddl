(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear e)
		(clear g)
		(clear h)
		(clear j)
		(clear k)
		(handempty)
		(on a b)
		(on b f)
		(on d i)
		(on f o)
		(on g c)
		(on i n)
		(on j d)
		(on k a)
		(on l m)
		(on o l)
		(ontable c)
		(ontable e)
		(ontable h)
		(ontable m)
		(ontable n)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)
