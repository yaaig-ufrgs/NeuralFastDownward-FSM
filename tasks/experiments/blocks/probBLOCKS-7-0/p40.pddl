(define (problem blocks-7-0)
(:domain blocks)
(:objects c f a b g d e )
(:init
	(clear e)
	(clear g)
	(handempty)
	(on a d)
	(on b a)
	(on e b)
	(on f c)
	(on g f)
	(ontable c)
	(ontable d)
)
(:goal (and (on a g) (on g d) (on d b) (on b c) (on c f) (on f e)))
)
