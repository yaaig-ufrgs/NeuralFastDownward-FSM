(define (problem blocks-12-0)
(:domain blocks)
(:objects i d b e k g a f c j l h )
		(:init
		(clear g)
		(clear j)
		(clear l)
		(holding k)
		(on a h)
		(on b d)
		(on d i)
		(on e b)
		(on f e)
		(on i c)
		(on j f)
		(on l a)
		(ontable c)
		(ontable g)
		(ontable h)
		)
(:goal (and (on i c) (on c b) (on b l) (on l d) (on d j) (on j e) (on e k)
            (on k f) (on f a) (on a h) (on h g)))
)
