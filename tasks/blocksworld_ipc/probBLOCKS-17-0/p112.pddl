(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear c)
		(clear d)
		(clear g)
		(clear j)
		(clear q)
		(holding e)
		(on a k)
		(on b m)
		(on c i)
		(on d a)
		(on f h)
		(on g p)
		(on h n)
		(on i b)
		(on j o)
		(on k f)
		(on p l)
		(ontable l)
		(ontable m)
		(ontable n)
		(ontable o)
		(ontable q)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
