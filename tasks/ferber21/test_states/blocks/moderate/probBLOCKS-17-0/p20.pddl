(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear a)
		(clear d)
		(clear o)
		(holding f)
		(on a n)
		(on b m)
		(on c j)
		(on d p)
		(on e l)
		(on h g)
		(on i b)
		(on j k)
		(on k i)
		(on l h)
		(on n q)
		(on p c)
		(on q e)
		(ontable g)
		(ontable m)
		(ontable o)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)