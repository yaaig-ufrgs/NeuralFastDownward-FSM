(define (problem blocks-12-0)
(:domain blocks)
(:objects i d b e k g a f c j l h )
		(:init
		(clear d)
		(clear e)
		(clear h)
		(clear i)
		(holding g)
		(on a l)
		(on d b)
		(on f k)
		(on h f)
		(on i c)
		(on j a)
		(on k j)
		(ontable b)
		(ontable c)
		(ontable e)
		(ontable l)
		)
(:goal (and (on i c) (on c b) (on b l) (on l d) (on d j) (on j e) (on e k)
            (on k f) (on f a) (on a h) (on h g)))
)
