(define (problem blocks-7-0)
(:domain blocks)
(:objects c f a b g d e )
(:init
	(clear b)
	(clear g)
	(handempty)
	(on a f)
	(on b e)
	(on d a)
	(on e d)
	(on f c)
	(ontable c)
	(ontable g)
)
(:goal (and (on a g) (on g d) (on d b) (on b c) (on c f) (on f e)))
)
