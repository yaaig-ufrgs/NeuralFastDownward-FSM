(define (problem blocks-7-0)
(:domain blocks)
(:objects c f a b g d e )
(:init
	(clear b)
	(clear c)
	(clear d)
	(clear e)
	(handempty)
	(on b g)
	(on e f)
	(on f a)
	(ontable a)
	(ontable c)
	(ontable d)
	(ontable g)
)
(:goal (and (on a g) (on g d) (on d b) (on b c) (on c f) (on f e)))
)
