(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear a)
		(clear c)
		(clear e)
		(clear l)
		(handempty)
		(on b f)
		(on d i)
		(on e j)
		(on f k)
		(on g n)
		(on h p)
		(on i g)
		(on j d)
		(on l q)
		(on m b)
		(on n o)
		(on p m)
		(on q h)
		(ontable a)
		(ontable c)
		(ontable k)
		(ontable o)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)