(define (problem blocks-12-0)
(:domain blocks)
(:objects i d b e k g a f c j l h )
		(:init
		(clear b)
		(clear c)
		(clear d)
		(clear j)
		(handempty)
		(on a h)
		(on b f)
		(on c i)
		(on h e)
		(on i l)
		(on j k)
		(on k g)
		(on l a)
		(ontable d)
		(ontable e)
		(ontable f)
		(ontable g)
		)
(:goal (and (on i c) (on c b) (on b l) (on l d) (on d j) (on j e) (on e k)
            (on k f) (on f a) (on a h) (on h g)))
)
