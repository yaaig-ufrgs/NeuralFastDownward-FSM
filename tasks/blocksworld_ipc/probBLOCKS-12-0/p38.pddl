(define (problem blocks-12-0)
(:domain blocks)
(:objects i d b e k g a f c j l h )
		(:init
		(clear b)
		(clear e)
		(clear h)
		(clear k)
		(handempty)
		(on a g)
		(on b a)
		(on d j)
		(on e d)
		(on h l)
		(on i c)
		(on k i)
		(on l f)
		(ontable c)
		(ontable f)
		(ontable g)
		(ontable j)
		)
(:goal (and (on i c) (on c b) (on b l) (on l d) (on d j) (on j e) (on e k)
            (on k f) (on f a) (on a h) (on h g)))
)
