(define (problem blocks-12-0)
(:domain blocks)
(:objects i d b e k g a f c j l h )
		(:init
		(clear a)
		(clear d)
		(clear l)
		(holding j)
		(on a c)
		(on b e)
		(on e k)
		(on g b)
		(on h f)
		(on i g)
		(on k h)
		(on l i)
		(ontable c)
		(ontable d)
		(ontable f)
		)
(:goal (and (on i c) (on c b) (on b l) (on l d) (on d j) (on j e) (on e k)
            (on k f) (on f a) (on a h) (on h g)))
)
