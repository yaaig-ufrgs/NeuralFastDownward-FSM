(define (problem blocks-12-0)
(:domain blocks)
(:objects i d b e k g a f c j l h )
		(:init
		(clear b)
		(clear j)
		(clear k)
		(clear l)
		(handempty)
		(on a f)
		(on b d)
		(on d i)
		(on e g)
		(on h a)
		(on i c)
		(on j h)
		(on l e)
		(ontable c)
		(ontable f)
		(ontable g)
		(ontable k)
		)
(:goal (and (on i c) (on c b) (on b l) (on l d) (on d j) (on j e) (on e k)
            (on k f) (on f a) (on a h) (on h g)))
)
