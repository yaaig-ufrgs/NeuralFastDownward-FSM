(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear f)
		(clear i)
		(clear j)
		(clear o)
		(clear q)
		(holding p)
		(on a k)
		(on b m)
		(on d c)
		(on e l)
		(on f e)
		(on h n)
		(on i b)
		(on j g)
		(on k d)
		(on o a)
		(on q h)
		(ontable c)
		(ontable g)
		(ontable l)
		(ontable m)
		(ontable n)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
