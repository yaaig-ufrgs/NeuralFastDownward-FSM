(define (problem blocks-12-0)
(:domain blocks)
(:objects i d b e k g a f c j l h )
		(:init
		(clear e)
		(clear g)
		(clear j)
		(clear l)
		(handempty)
		(on b d)
		(on d i)
		(on e b)
		(on g a)
		(on i c)
		(on j k)
		(on k f)
		(on l h)
		(ontable a)
		(ontable c)
		(ontable f)
		(ontable h)
		)
(:goal (and (on i c) (on c b) (on b l) (on l d) (on d j) (on j e) (on e k)
            (on k f) (on f a) (on a h) (on h g)))
)
