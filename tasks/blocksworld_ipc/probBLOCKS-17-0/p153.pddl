(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear d)
		(clear f)
		(clear j)
		(clear l)
		(clear o)
		(holding c)
		(on b m)
		(on d a)
		(on e q)
		(on f g)
		(on g h)
		(on h n)
		(on i b)
		(on j e)
		(on k p)
		(on o k)
		(on p i)
		(ontable a)
		(ontable l)
		(ontable m)
		(ontable n)
		(ontable q)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
