(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear a)
		(clear d)
		(clear e)
		(clear g)
		(clear h)
		(clear j)
		(handempty)
		(on a k)
		(on b m)
		(on c o)
		(on d c)
		(on e q)
		(on g f)
		(on h l)
		(on i b)
		(on j i)
		(on l n)
		(on q p)
		(ontable f)
		(ontable k)
		(ontable m)
		(ontable n)
		(ontable o)
		(ontable p)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
