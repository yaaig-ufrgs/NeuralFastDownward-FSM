(define (problem blocks-7-0)
(:domain blocks)
(:objects c f a b g d e )
(:init
	(clear c)
	(clear e)
	(clear g)
	(handempty)
	(on a d)
	(on b a)
	(on c f)
	(on e b)
	(ontable d)
	(ontable f)
	(ontable g)
)
(:goal (and (on a g) (on g d) (on d b) (on b c) (on c f) (on f e)))
)
