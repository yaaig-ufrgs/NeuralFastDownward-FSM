(define (problem blocks-12-0)
(:domain blocks)
(:objects i d b e k g a f c j l h )
		(:init
		(clear b)
		(clear c)
		(clear d)
		(clear f)
		(holding e)
		(on a g)
		(on b a)
		(on c i)
		(on d j)
		(on i k)
		(on k l)
		(on l h)
		(ontable f)
		(ontable g)
		(ontable h)
		(ontable j)
		)
(:goal (and (on i c) (on c b) (on b l) (on l d) (on d j) (on j e) (on e k)
            (on k f) (on f a) (on a h) (on h g)))
)
