(define (problem blocks-12-0)
(:domain blocks)
(:objects i d b e k g a f c j l h )
		(:init
		(clear c)
		(clear e)
		(clear i)
		(holding b)
		(on a j)
		(on c h)
		(on d a)
		(on e g)
		(on g k)
		(on h d)
		(on k l)
		(on l f)
		(ontable f)
		(ontable i)
		(ontable j)
		)
(:goal (and (on i c) (on c b) (on b l) (on l d) (on d j) (on j e) (on e k)
            (on k f) (on f a) (on a h) (on h g)))
)
