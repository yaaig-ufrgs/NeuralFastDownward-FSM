(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear b)
		(clear c)
		(clear g)
		(clear j)
		(clear o)
		(holding m)
		(on a q)
		(on b p)
		(on c d)
		(on d e)
		(on e f)
		(on g a)
		(on h n)
		(on i l)
		(on j k)
		(on k i)
		(on q h)
		(ontable f)
		(ontable l)
		(ontable n)
		(ontable o)
		(ontable p)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
