(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear e)
		(clear f)
		(clear h)
		(clear k)
		(handempty)
		(on a j)
		(on b m)
		(on c o)
		(on d c)
		(on e n)
		(on f l)
		(on g q)
		(on h p)
		(on i b)
		(on j i)
		(on l g)
		(on n d)
		(on q a)
		(ontable k)
		(ontable m)
		(ontable o)
		(ontable p)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
