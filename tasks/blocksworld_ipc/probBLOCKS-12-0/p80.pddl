(define (problem blocks-12-0)
(:domain blocks)
(:objects i d b e k g a f c j l h )
		(:init
		(clear i)
		(clear k)
		(handempty)
		(on a l)
		(on c a)
		(on d j)
		(on e b)
		(on f c)
		(on h f)
		(on i d)
		(on j e)
		(on k h)
		(on l g)
		(ontable b)
		(ontable g)
		)
(:goal (and (on i c) (on c b) (on b l) (on l d) (on d j) (on j e) (on e k)
            (on k f) (on f a) (on a h) (on h g)))
)
