(define (problem blocks-12-0)
(:domain blocks)
(:objects i d b e k g a f c j l h )
		(:init
		(clear d)
		(clear f)
		(clear g)
		(clear i)
		(holding k)
		(on a h)
		(on b e)
		(on e l)
		(on f a)
		(on g b)
		(on h j)
		(on i c)
		(ontable c)
		(ontable d)
		(ontable j)
		(ontable l)
		)
(:goal (and (on i c) (on c b) (on b l) (on l d) (on d j) (on j e) (on e k)
            (on k f) (on f a) (on a h) (on h g)))
)
