(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear a)
		(clear c)
		(clear d)
		(handempty)
		(on a e)
		(on b j)
		(on c f)
		(on d g)
		(on f h)
		(on g n)
		(on h o)
		(on j i)
		(on k b)
		(on l k)
		(on n l)
		(on o m)
		(ontable e)
		(ontable i)
		(ontable m)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)
