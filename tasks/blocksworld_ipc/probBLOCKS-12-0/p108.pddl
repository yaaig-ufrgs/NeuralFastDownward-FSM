(define (problem blocks-12-0)
(:domain blocks)
(:objects i d b e k g a f c j l h )
		(:init
		(clear c)
		(clear d)
		(clear h)
		(clear j)
		(handempty)
		(on a b)
		(on b g)
		(on c l)
		(on e f)
		(on g k)
		(on h i)
		(on j a)
		(on k e)
		(ontable d)
		(ontable f)
		(ontable i)
		(ontable l)
		)
(:goal (and (on i c) (on c b) (on b l) (on l d) (on d j) (on j e) (on e k)
            (on k f) (on f a) (on a h) (on h g)))
)
