(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear b)
		(clear g)
		(clear j)
		(handempty)
		(on a e)
		(on b i)
		(on c m)
		(on d f)
		(on e h)
		(on f l)
		(on g d)
		(on h n)
		(on i c)
		(on j o)
		(on l p)
		(on m a)
		(on o q)
		(on q k)
		(ontable k)
		(ontable n)
		(ontable p)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
