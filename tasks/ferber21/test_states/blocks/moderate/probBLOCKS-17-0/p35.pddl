(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear f)
		(clear h)
		(clear j)
		(clear k)
		(clear m)
		(clear n)
		(handempty)
		(on a g)
		(on c o)
		(on d c)
		(on e q)
		(on f p)
		(on g d)
		(on h b)
		(on j e)
		(on l a)
		(on m i)
		(on q l)
		(ontable b)
		(ontable i)
		(ontable k)
		(ontable n)
		(ontable o)
		(ontable p)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
