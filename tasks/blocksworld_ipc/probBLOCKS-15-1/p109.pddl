(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear b)
		(clear c)
		(clear d)
		(clear f)
		(handempty)
		(on a h)
		(on b m)
		(on c j)
		(on d g)
		(on e k)
		(on f o)
		(on g e)
		(on h n)
		(on j l)
		(on l i)
		(on m a)
		(ontable i)
		(ontable k)
		(ontable n)
		(ontable o)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)
