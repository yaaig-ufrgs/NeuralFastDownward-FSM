(define (problem blocks-12-0)
(:domain blocks)
(:objects i d b e k g a f c j l h )
		(:init
		(clear c)
		(clear d)
		(clear e)
		(clear i)
		(clear k)
		(handempty)
		(on b l)
		(on c j)
		(on e f)
		(on g h)
		(on h a)
		(on j b)
		(on l g)
		(ontable a)
		(ontable d)
		(ontable f)
		(ontable i)
		(ontable k)
		)
(:goal (and (on i c) (on c b) (on b l) (on l d) (on d j) (on j e) (on e k)
            (on k f) (on f a) (on a h) (on h g)))
)
