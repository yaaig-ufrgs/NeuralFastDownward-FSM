(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear d)
		(clear e)
		(clear l)
		(clear n)
		(clear q)
		(holding p)
		(on a j)
		(on b m)
		(on c o)
		(on d c)
		(on e h)
		(on f k)
		(on g f)
		(on i b)
		(on j i)
		(on l g)
		(on q a)
		(ontable h)
		(ontable k)
		(ontable m)
		(ontable n)
		(ontable o)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)